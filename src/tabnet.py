# src/tabnet.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Dataset  (same interface as fnn.py / resfnn.py)
# ---------------------------------------------------------------------------

class PlayCallDataset(Dataset):
    """Wraps a feature df and target series as a PyTorch dataset."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# TabNet building blocks
# ---------------------------------------------------------------------------

class GLUBlock(nn.Module):
    """
    Gated Linear Unit block used inside each TabNet step.

    A shared layer and a step-specific layer are combined with a
    GLU activation, which multiplies a linear transform by a sigmoid gate.
    This lets the network selectively suppress or amplify information at
    each position, giving it expressive power without deep stacking.

    Architecture (one half):
        x → Linear(in → 2·out) → BatchNorm → GLU → √0.5 residual scale
    """

    def __init__(self, in_features: int, out_features: int, fc_shared: nn.Linear):
        super().__init__()
        # shared sub-layer: reused across all steps to regularise representations
        self.fc_shared = fc_shared
        self.bn_shared = nn.BatchNorm1d(out_features * 2)

        # step-specific sub-layer: learns per-step refinements on top of the shared layer
        self.fc_step = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn_step = nn.BatchNorm1d(out_features * 2)

        self.out_features = out_features
        # √0.5 scale keeps variance stable when adding residual to the next block
        self.scale = (0.5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shared path
        h_shared = self.bn_shared(self.fc_shared(x))
        # step-specific path
        h_step = self.bn_step(self.fc_step(x))
        # element-wise sum and GLU: split in half, then multiply by sigmoid gate
        h = h_shared + h_step
        return torch.nn.functional.glu(h, dim=-1) * self.scale


class TabNetStep(nn.Module):
    """
    One attentive step in TabNet.

    Each step consists of:
        1. Attentive Transformer – a sparse softmax (entmax / sparsemax) mask
           that selects which features the step should focus on.
        2. Feature Transformer – two stacked GLU blocks that transform the
           masked features into a latent representation.

    The attention mask is guided by a prior-scale that accumulates how much
    each feature has been used in previous steps, discouraging redundancy.

    Parameters
        n_features   : number of input features (after scaling)
        n_d          : width of the decision-step output (passed to classifier head)
        n_a          : width of the attentive-transformer output
        fc_shared_1  : shared Linear for the first GLU block (cross-step weight sharing)
        fc_shared_2  : shared Linear for the second GLU block
        momentum     : BatchNorm momentum (default 0.02 — TabNet paper default)
    """

    def __init__(
        self,
        n_features: int,
        n_d: int,
        n_a: int,
        fc_shared_1: nn.Linear,
        fc_shared_2: nn.Linear,
        momentum: float = 0.02,
    ):
        super().__init__()

        # Attentive Transformer: maps prior-scaled features → attention weights
        self.fc_att  = nn.Linear(n_a, n_features, bias=False)
        self.bn_att  = nn.BatchNorm1d(n_features, momentum=momentum)

        # Feature Transformer: two GLU blocks
        self.glu1 = GLUBlock(n_features, n_d + n_a, fc_shared_1)
        self.glu2 = GLUBlock(n_d + n_a, n_d + n_a, fc_shared_2)

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
            x            : scaled input features  (batch × n_features)
            prior_scales : accumulated usage mask  (batch × n_features)
            h_prev       : previous step's attentive output for carry-over (batch × n_a)

        Returns
            h_out        : decision output for this step  (batch × n_d)
            h_a          : attentive state passed to the next step (batch × n_a)
            alpha        : sparse attention weights for this step (batch × n_features)
        """
        # --- attention mask ---
        # project prior-weighted carry-over to feature space
        att_logits = self.bn_att(self.fc_att(h_prev))
        # multiply by prior scales to penalise already-used features
        att_logits = att_logits * prior_scales
        # sparsemax: projects onto the probability simplex while encouraging zeros
        alpha = self._sparsemax(att_logits)

        # apply mask to input
        masked_x = alpha * x

        # --- feature transformer ---
        h = self.glu1(masked_x)
        h = self.glu2(h)

        # split into decision output (n_d) and attentive carry-over (n_a)
        n_d = h.shape[-1] // 2  # glu2 output is n_d + n_a
        h_out = h[:, :n_d]
        h_a   = h[:, n_d:]

        return h_out, h_a, alpha

    @staticmethod
    def _sparsemax(z: torch.Tensor) -> torch.Tensor:
        """
        Sparsemax activation (Martins & Astudillo, 2016).

        Projects z onto the probability simplex, with many outputs set to
        exactly zero — sparser and sharper than softmax.
        """
        # sort descending along the feature dimension
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        # cumulative sum for projection bound
        z_cumsum = torch.cumsum(z_sorted, dim=-1)
        # thresholding index k(z)
        k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # last True index per row
        k_z = z_check.sum(dim=-1, keepdim=True)
        tau = (z_cumsum.gather(-1, k_z - 1) - 1) / k_z.float()
        return torch.clamp(z - tau, min=0)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TabNet(nn.Module):
    """
    TabNet for binary play-call classification.

    TabNet processes tabular data with sequential attentive steps.
    Each step learns a sparse mask over the input features, selectively
    focusing on the most relevant subset.  The outputs of all steps are
    summed and fed to a single linear head for the final prediction.

    Key advantages over MLP-style networks
        - Interpretability  : per-step feature masks show which inputs mattered
        - Sparse attention  : only a subset of features is used per step (sparsity)
        - No hand-crafted embeddings required for heterogeneous tabular features

    Architecture
        Shared Feature Transformer (GLU blocks, cross-step weight sharing)
        N_steps × TabNetStep:
            Attentive Transformer (sparse feature selection)
            Feature Transformer   (two GLU blocks)
        Sum of decision outputs  → ReLU → Linear(n_d → 1)

    Parameters
        n_features      : number of input features (inferred from X_train)
        n_d             : width of decision-step output embeddings (default 32)
        n_a             : width of attentive-transformer hidden state (default 32)
        n_steps         : number of sequential attentive steps (default 4)
        gamma           : sparsity regularisation coefficient for prior scales (default 1.5)
        momentum        : BatchNorm momentum (default 0.02)

    Improvements over ResFNN
        - Attentive feature selection   — learns which features matter per step
        - Sparsemax masks               — explicit feature sparsity, better generalisation
        - Sequential multi-step design  — refines representation iteratively
        - Interpretable attention masks — can be extracted for feature importance
        - Sparsity regularisation loss  — penalises overuse of all features at once
    """

    def __init__(
        self,
        n_features: int,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 4,
        gamma: float = 1.5,
        momentum: float = 0.02,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_d        = n_d
        self.n_a        = n_a
        self.n_steps    = n_steps
        self.gamma      = gamma

        # initial batch normalisation of raw inputs
        self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)

        # shared linear layers (weight-tied across all steps for regularisation)
        step_input_dim = n_features
        self.fc_shared_1 = nn.Linear(step_input_dim, (n_d + n_a) * 2, bias=False)
        self.fc_shared_2 = nn.Linear(n_d + n_a,      (n_d + n_a) * 2, bias=False)

        # attentive steps
        self.steps = nn.ModuleList([
            TabNetStep(
                n_features=n_features,
                n_d=n_d,
                n_a=n_a,
                fc_shared_1=self.fc_shared_1,
                fc_shared_2=self.fc_shared_2,
                momentum=momentum,
            )
            for _ in range(n_steps)
        ])

        # output head: aggregate step outputs → scalar logit
        self.output_head = nn.Linear(n_d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
            x : raw input features (batch × n_features)

        Returns
            logits     : un-normalised output (batch,) — use BCEWithLogitsLoss
            sparsity   : mean entropy across steps for sparsity regularisation
        """
        # normalise inputs
        x = self.initial_bn(x)

        # initialise prior scales to ones (no feature has been used yet)
        prior_scales = torch.ones(x.shape[0], self.n_features, device=x.device)

        # initialise attentive carry-over (h_a) as zeros
        h_a = torch.zeros(x.shape[0], self.n_a, device=x.device)

        decision_outputs = []
        sparsity_loss    = torch.zeros(1, device=x.device)

        for step in self.steps:
            h_out, h_a, alpha = step(x, prior_scales, h_a)
            decision_outputs.append(h_out)

            # update prior scales: features used now cost more in future steps
            prior_scales = prior_scales * (self.gamma - alpha)

            # sparsity regularisation: entropy of the attention distribution
            # low entropy → sharp mask → good sparsity
            sparsity_loss += (-alpha * torch.log(alpha + 1e-15)).sum(dim=-1).mean()

        # aggregate: sum step outputs, apply ReLU, project to scalar
        agg = torch.stack(decision_outputs, dim=0).sum(dim=0)
        agg = torch.relu(agg)
        logits = self.output_head(agg).squeeze(1)

        return logits, sparsity_loss / self.n_steps

    def get_feature_importances(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute per-feature importance as the sum of attention weights
        across all steps, averaged over the batch.

        Returns
            importances : ndarray of shape (n_features,) — higher = more used
        """
        self.eval()
        x = self.initial_bn(x)
        prior_scales = torch.ones(x.shape[0], self.n_features, device=x.device)
        h_a          = torch.zeros(x.shape[0], self.n_a, device=x.device)

        total_alpha = torch.zeros(x.shape[0], self.n_features, device=x.device)
        with torch.no_grad():
            for step in self.steps:
                _, h_a, alpha = step(x, prior_scales, h_a)
                prior_scales  = prior_scales * (self.gamma - alpha)
                total_alpha  += alpha

        return total_alpha.mean(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_tabnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_split: float = 0.1,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 2e-3,
    patience: int = 20,
    n_d: int = 32,
    n_a: int = 32,
    n_steps: int = 4,
    gamma: float = 1.5,
    lambda_sparse: float = 1e-4,
    weight_decay: float = 1e-5,
    random_state: int = 42,
) -> tuple["TabNet", dict, StandardScaler]:
    """
    Train the TabNet with early stopping, LR scheduling, and sparsity loss.

    Key design choices
        - AdamW with mild weight_decay        — decoupled L2 regularisation
        - ReduceLROnPlateau                   — halves LR after 7 stagnant epochs
        - BCEWithLogitsLoss + pos_weight      — handles class imbalance
        - Sparsity loss (lambda_sparse)       — encourages focused feature masks
        - Best weights restored after training regardless of early stopping

    Parameters
        X_train        : training feature DataFrame (output of get_X_y)
        y_train        : training target Series (1 = pass, 0 = run)
        val_split      : fraction used for internal validation (default 0.1)
        epochs         : maximum training epochs (default 200)
        batch_size     : mini-batch size (default 512)
        lr             : initial AdamW learning rate (default 2e-3)
        patience       : early stopping patience in epochs (default 20)
        n_d            : decision-step embedding width (default 32)
        n_a            : attentive-transformer hidden width (default 32)
        n_steps        : number of sequential attentive steps (default 4)
        gamma          : prior-scale sparsity coefficient (default 1.5)
        lambda_sparse  : weight for sparsity regularisation loss (default 1e-4)
        weight_decay   : L2 regularisation strength for AdamW (default 1e-5)
        random_state   : seed for reproducibility

    Returns
        model   : trained TabNet with best validation weights restored
        history : dict with train_loss, val_loss, and lr per epoch
        scaler  : fitted StandardScaler (needed by TabNetWrapper)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # scale features — fit only on training data to prevent leakage
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # build dataset and split
    full_dataset = PlayCallDataset(X_train, y_train)
    val_size     = int(len(full_dataset) * val_split)
    train_size   = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # pos_weight = n_negative / n_positive — upweights the minority class in loss
    y_arr      = y_train.values
    n_pos      = y_arr.sum()
    n_neg      = len(y_arr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # model, loss, optimiser, scheduler
    n_features = X_train.shape[1]
    model      = TabNet(
        n_features=n_features,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
    )
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,       # halve LR on plateau
        patience=7,       # wait 7 epochs before reducing
        min_lr=1e-6,
    )

    history           = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):

        # --- training pass ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, sparsity = model(X_batch)
            # classification loss + sparsity regularisation
            loss = criterion(logits, y_batch) + lambda_sparse * sparsity
            loss.backward()
            # gradient clipping prevents rare exploding-gradient issues
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- validation pass ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits, sparsity = model(X_batch)
                loss = criterion(logits, y_batch) + lambda_sparse * sparsity
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # step scheduler on val_loss
        scheduler.step(val_loss)

        # early stopping + best weight tracking
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve == patience:
            print(
                f"  Epoch {epoch:>3} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"no_improve: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print(f"\n[train_tabnet] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_weights)
    print(f"[train_tabnet] Training complete. Best val_loss: {best_val_loss:.4f}")

    return model, history, scaler


# ---------------------------------------------------------------------------
# Wrapper  (sklearn-compatible interface for evaluate_model / plot functions)
# ---------------------------------------------------------------------------

class TabNetWrapper:
    """
    Sklearn-style wrapper around a trained TabNet.

    The model outputs raw logits; this wrapper applies sigmoid to convert
    them to probabilities and a threshold to produce hard predictions.

    Usage
        wrapper = TabNetWrapper(model, scaler)
        metrics = evaluate_model(wrapper, X_test, y_test, "TabNet", "maxi")
    """

    def __init__(self, model: TabNet, scaler: StandardScaler, threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            # sigmoid converts logits → probabilities
            proba_pass = torch.sigmoid(logits).numpy()
        proba_run = 1 - proba_pass
        return np.column_stack([proba_run, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
