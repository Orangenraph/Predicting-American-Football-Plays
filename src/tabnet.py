import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlayCallDataset(Dataset):
    """
    PyTorch Dataset wrapper for feature dataframes and targets.

    Loads tensors directly onto the target device if the dataset fits in 
    GPU memory, bypassing per-batch CPU-to-GPU transfer overhead.

    Parameters
        X      : feature DataFrame
        y      : target Series
        device : target torch.device
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, device: torch.device | None = None):
        # determine target execution device
        dev = device if device is not None else torch.device("cpu")
        
        # convert input pandas data structures into PyTorch tensors and move to device
        self.X      = torch.tensor(X.values, dtype=torch.float32).to(dev)
        self.y      = torch.tensor(y.values, dtype=torch.float32).to(dev)
        
        # track execution location for pipeline decisions
        self.on_gpu = dev.type == "cuda"

    def __len__(self):
        # return the total number of records in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # retrieve features and corresponding label at specified index
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# TabNet building blocks
# ---------------------------------------------------------------------------

class GLUBlock(nn.Module):
    """
    Gated Linear Unit block used inside each TabNet step.

    Uses Ghost Batch Normalization (GBN) instead of standard BN. GBN splits 
    the batch into virtual sub-batches to stabilize training with large batches 
    and act as a regularizer.

    Parameters
        in_features        : input dimensions
        out_features       : output dimensions of the GLU block
        fc_shared          : shared linear layer instance
        virtual_batch_size : batch size for Ghost BN
        momentum           : momentum for batch normalization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_shared: nn.Linear,
        virtual_batch_size: int = 256,
        momentum: float = 0.02,
    ):
        super().__init__()
        # assign shared linear projection and set virtual batch size
        self.fc_shared          = fc_shared
        self.virtual_batch_size = virtual_batch_size

        # initialize batch normalization for the shared path
        self.bn_shared = nn.BatchNorm1d(out_features * 2, momentum=momentum)
        
        # define step-specific linear projection and batch normalization
        self.fc_step   = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn_step   = nn.BatchNorm1d(out_features * 2, momentum=momentum)

        # store output dimensions and scaling factor to control variance growth
        self.out_features = out_features
        self.scale        = (0.5 ** 0.5)

    def _ghost_bn(self, bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization on virtual sub-batches."""
        # retrieve current batch size and virtual sub-batch threshold
        B = x.shape[0]
        vbs = self.virtual_batch_size

        # bypass sub-batch normalization if in evaluation mode or batch is too small
        if not self.training or B <= vbs:
            return bn(x)

        # partition batch into virtual sub-batches
        chunks  = x.split(vbs, dim=0)
        
        # normalize each sub-batch individually using shared running statistics
        normed  = [bn(c) for c in chunks]
        
        # reconstruct the unified batch representation
        return torch.cat(normed, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process input through shared and step-specific projections with ghost BN
        h_shared = self._ghost_bn(self.bn_shared, self.fc_shared(x))
        h_step   = self._ghost_bn(self.bn_step,   self.fc_step(x))
        
        # aggregate projected representations
        h        = h_shared + h_step
        
        # apply gated linear unit activation followed by scaling
        return torch.nn.functional.glu(h, dim=-1) * self.scale


class TabNetStep(nn.Module):
    """
    Single sequential selection step in TabNet.

    Includes attention dropout in the Attentive Transformer to prevent 
    the attention mask from collapsing onto too few features.

    Parameters
        n_features         : number of input features
        n_d                : dimensionality of decision prediction representation
        n_a                : dimensionality of attention transformer representation
        fc_shared_1        : first shared linear layer
        fc_shared_2        : second shared linear layer
        momentum           : momentum for batch normalization
        virtual_batch_size : batch size for Ghost BN
        att_dropout        : dropout rate for attention mask logits
    """

    def __init__(
        self,
        n_features: int,
        n_d: int,
        n_a: int,
        fc_shared_1: nn.Linear,
        fc_shared_2: nn.Linear,
        momentum: float = 0.02,
        virtual_batch_size: int = 256,
        att_dropout: float = 0.1,
    ):
        super().__init__()

        # initialize linear and batchnorm steps for attention computation
        self.fc_att     = nn.Linear(n_a, n_features, bias=False)
        self.bn_att     = nn.BatchNorm1d(n_features, momentum=momentum)
        
        # define attention dropout to prevent over-reliance on limited features
        self.att_drop   = nn.Dropout(p=att_dropout)

        # construct feature transformer block using sequence of two GLU components
        self.glu1 = GLUBlock(n_features,   n_d + n_a, fc_shared_1, virtual_batch_size, momentum)
        self.glu2 = GLUBlock(n_d + n_a,    n_d + n_a, fc_shared_2, virtual_batch_size, momentum)

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # transform previous steps state into attention space and normalize
        att_logits = self.bn_att(self.fc_att(h_prev))
        
        # apply attention dropout to randomly mask parts of decision logits
        att_logits = self.att_drop(att_logits)
        
        # adjust logits using scale history and project via sparsemax to generate attention mask
        att_logits = att_logits * prior_scales
        alpha      = self._sparsemax(att_logits)

        # filter input feature representation using computed mask
        masked_x = alpha * x

        # process filtered features through sequential GLU blocks
        h     = self.glu1(masked_x)
        h     = self.glu2(h)
        
        # partition feature transformer output into decision and attention channels
        n_d   = h.shape[-1] // 2
        h_out = h[:, :n_d]
        h_a   = h[:, n_d:]

        return h_out, h_a, alpha

    @staticmethod
    def _sparsemax(z: torch.Tensor) -> torch.Tensor:
        """Sparsemax activation function (Martins & Astudillo, 2016)."""
        # sort logits in descending order for cumulative distribution calculations
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        z_cumsum    = torch.cumsum(z_sorted, dim=-1)
        
        # define coordinate index mapping array
        k           = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
        
        # identify threshold indicators where conditional mass is strictly positive
        z_check     = 1 + k * z_sorted > z_cumsum
        k_z         = z_check.sum(dim=-1, keepdim=True)
        
        # compute probability threshold offset parameter
        tau         = (z_cumsum.gather(-1, k_z - 1) - 1) / k_z.float()
        
        # clip results to maintain correct output support
        return torch.clamp(z - tau, min=0)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TabNet(nn.Module):
    """
    TabNet architecture adapted for binary play-call classification.

    Design adjustments from the baseline implementation:
        - Increased step capacities (n_d, n_a) to mitigate early underfitting.
        - Higher sequential depth (n_steps) to resolve complex tabular patterns.
        - Lower gamma value to encourage feature reuse across steps.
        - Dropout layers added to the decision head to manage train-validation gap.

    Parameters
        n_features         : number of input features
        n_d                : output dimension for prediction representation
        n_a                : output dimension for attention mask
        n_steps            : count of sequential decision steps
        gamma              : scaling coefficient for prioritizing unused features
        momentum           : batch normalization momentum
        virtual_batch_size : batch size for Ghost BN
        att_dropout        : dropout rate for attention mask logits
        final_dropout      : dropout rate before the output head
    """

    def __init__(
        self,
        n_features:         int,
        n_d:                int,
        n_a:                int,
        n_steps:            int,
        gamma:              float,
        momentum:           float,
        virtual_batch_size: int,
        att_dropout:        float,
        final_dropout:      float,
    ):
        super().__init__()

        # store global execution configurations
        self.n_features = n_features
        self.n_d        = n_d
        self.n_a        = n_a
        self.n_steps    = n_steps
        self.gamma      = gamma

        # perform baseline normalisation on raw input features
        self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)

        # construct shared linear structures to reuse weights across sequential steps
        self.fc_shared_1 = nn.Linear(n_features,  (n_d + n_a) * 2, bias=False)
        self.fc_shared_2 = nn.Linear(n_d + n_a,   (n_d + n_a) * 2, bias=False)

        # construct sequential decision steps
        self.steps = nn.ModuleList([
            TabNetStep(
                n_features=n_features,
                n_d=n_d,
                n_a=n_a,
                fc_shared_1=self.fc_shared_1,
                fc_shared_2=self.fc_shared_2,
                momentum=momentum,
                virtual_batch_size=virtual_batch_size,
                att_dropout=att_dropout,
            )
            for _ in range(n_steps)
        ])

        # construct final output decision structure
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.output_head   = nn.Linear(n_d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # normalize raw input values
        x = self.initial_bn(x)

        # initialize scaling metrics, state trackers, and sparsity loss containers
        prior_scales     = torch.ones(x.shape[0], self.n_features, device=x.device)
        h_a              = torch.zeros(x.shape[0], self.n_a, device=x.device)
        decision_outputs = []
        sparsity_loss    = torch.zeros(1, device=x.device)

        # execute sequential selection steps
        for step in self.steps:
            h_out, h_a, alpha = step(x, prior_scales, h_a)
            decision_outputs.append(h_out)
            
            # update tracking status of unused features and compile entropy metrics
            prior_scales = prior_scales * (self.gamma - alpha)
            sparsity_loss += (-alpha * torch.log(alpha + 1e-15)).sum(dim=-1).mean()

        # synthesize outcomes across steps
        agg    = torch.stack(decision_outputs, dim=0).sum(dim=0)
        agg    = torch.relu(agg)
        
        # apply regularization prior to prediction mapping
        agg    = self.final_dropout(agg)
        logits = self.output_head(agg).squeeze(1)

        # return estimated logits along with averaged entropy loss of attention weights
        return logits, sparsity_loss / self.n_steps

    def get_feature_importances(self, x: torch.Tensor) -> np.ndarray:
        """Compute aggregated attention weights across steps, averaged over the batch."""
        # switch state to evaluation mode and apply initial normalisation
        self.eval()
        x            = self.initial_bn(x)
        
        # initialize scale tracking configurations
        prior_scales = torch.ones(x.shape[0], self.n_features, device=x.device)
        h_a          = torch.zeros(x.shape[0], self.n_a, device=x.device)
        total_alpha  = torch.zeros(x.shape[0], self.n_features, device=x.device)

        # accumulate selection metrics across all layers without tracking gradients
        with torch.no_grad():
            for step in self.steps:
                _, h_a, alpha = step(x, prior_scales, h_a)
                prior_scales  = prior_scales * (self.gamma - alpha)
                total_alpha  += alpha

        # compile average feature contribution scores across batch
        return total_alpha.mean(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_tabnet(
    X_train, y_train,
    # Data
    val_split:          float,
    random_state:       int,
    # Architecture
    n_d:                int,
    n_a:                int,
    n_steps:            int,
    gamma:              float,
    momentum:           float,
    virtual_batch_size: int,
    att_dropout:        float,
    final_dropout:      float,
    # Training
    epochs:             int,
    batch_size:         int,
    lr:                 float,
    patience:           int,
    lambda_sparse:      float,
    weight_decay:       float,
    # Scheduler
    grad_clip:          float,
    T_0:                int,
    T_mult:             int,
    eta_min:            float,
) -> tuple["TabNet", dict, StandardScaler]:
    """
    Train TabNet using early stopping, cosine annealing, and sparsity objectives.

    Hyperparameter updates:
        - Adjusted lambda_sparse for stronger sparsity enforcement.
        - Increased weight decay to limit overfitting risks.
        - Scaled batch size to optimize GPU batch operations.
        - Extended patience window to allow deeper models to converge.
    """
    # set deterministic computational constraints
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # configure computational target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tabnet] Using device: {device}")

    # calculate standardization metrics only on the training feature distribution
    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # instantiate the master dataset directly on the target execution device
    preload_device = device if device.type == "cuda" else None
    full_dataset   = PlayCallDataset(X_scaled, y_train, device=preload_device)

    # split the data into training and validation sets
    val_size     = int(len(full_dataset) * val_split)
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    # optimize data loading process based on execution device status
    nw = 0 if (device.type == "cuda") else 2
    pw = device.type != "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=pw, num_workers=nw,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=pw, num_workers=nw,
        persistent_workers=(nw > 0),
    )

    # compute balancing weights to manage class imbalance
    y_arr      = y_train.values
    n_pos      = y_arr.sum()
    n_neg      = len(y_arr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    # instantiate model on target device
    n_features = X_train.shape[1]
    model = TabNet(
        n_features=n_features,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        virtual_batch_size=virtual_batch_size,
        att_dropout=att_dropout,
        final_dropout=final_dropout,
    ).to(device)

    # set loss objective, optimizer, and cyclic learning rate decay rules
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )

    # configure validation checkpoint storage parameters to minimize peak system memory usage
    best_ckpt  = os.path.join(tempfile.gettempdir(), "tabnet_best.pt")
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):

        # --- Training phase ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            # perform system memory copies only if data is not preloaded on target device
            if device.type != "cuda":
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

            # optimize weight configurations
            optimizer.zero_grad()
            logits, sparsity = model(X_batch)
            loss = criterion(logits, y_batch) + lambda_sparse * sparsity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # step learning rate decay cycle
        scheduler.step(epoch - 1)

        # --- Validation phase ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # transfer validation batches to device if required
                if device.type != "cuda":
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                logits, sparsity = model(X_batch)
                loss = criterion(logits, y_batch) + lambda_sparse * sparsity
                val_losses.append(loss.item())

        # calculate average loss values and store tracking metrics
        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # apply validation checks to determine checkpoint updates
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        # print optimization updates at set intervals
        if epoch % 10 == 0 or no_improve == patience:
            print(
                f"  Epoch {epoch:>3} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"no_improve: {no_improve}/{patience}"
            )

        # terminate training if validation progress plateaus
        if no_improve >= patience:
            print(f"\n[train_tabnet] Early stopping at epoch {epoch}.")
            break

    # restore best performing parameter weights
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[train_tabnet] Training complete. Best val_loss: {best_val:.4f}")

    return model, history, scaler


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TabNetWrapper:
    """
    Sklearn-compatible interface wrapper for PyTorch TabNet model.

    Simplifies calls from pipeline operations, model evaluations, 
    and plotting workflows.
    """

    def __init__(self, model: TabNet, scaler: StandardScaler, threshold: float = 0.5):
        # assign execution wrapper attributes
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # transition model to evaluation state and fetch system device target
        self.model.eval()
        device   = next(self.model.parameters()).device
        
        # scale features and map to PyTorch tensors on the proper device
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)

        # generate raw predictions and map them to standard probability distributions
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            proba_pass = torch.sigmoid(logits).cpu().numpy()

        # structure predicted probabilities as a standard two-class matrix
        return np.column_stack([1 - proba_pass, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # fetch probability estimates of positive target outcomes
        proba = self.predict_proba(X)[:, 1]
        
        # map probability scores to binary labels using set threshold limits
        return (proba >= self.threshold).astype(int)