# src/resfnn.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Dataset  (reused from fnn.py — identical interface)
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
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    A single residual block:
        x → Linear(in → out) → BN → ReLU → Dropout
          → Linear(out → out) → BN
          + skip(x projected to out if dims differ)
          → ReLU

    Batch Normalisation is applied before activation (pre-activation style).
    The skip connection stabilises gradients and allows the network to learn
    incremental refinements rather than full transformations at each layer.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )

        # Project input to out_features if dimensions differ so the skip
        # addition is always valid
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ResFNN(nn.Module):
    """
    Residual Feedforward Neural Network for binary play-call classification.

    Architecture
        Input projection : Linear(n_features → 256) → BN → ReLU → Dropout(0.3)
        ResidualBlock    : 256 → 128  (projects skip)
        ResidualBlock    : 128 → 64   (projects skip)
        Output head      : Linear(64 → 1) → Sigmoid

    Improvements over the baseline FNN
        - Residual (skip) connections  — stabilise gradients, ease optimisation
        - Batch Normalisation          — normalises layer inputs, speeds convergence
        - AdamW optimiser              — decoupled weight decay (L2 regularisation)
        - ReduceLROnPlateau scheduler  — reduces LR when val_loss stagnates
        - Pos-weight in BCEWithLogits  — handles class imbalance automatically
        - Longer patience (15)         — gives the scheduler time to act
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()

        # initial projection — expands the small feature space into a richer
        # representation before the residual blocks
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_block1 = ResidualBlock(256, 128, dropout=dropout)
        self.res_block2 = ResidualBlock(128, 64, dropout=max(dropout - 0.1, 0.1))

        # output head — no Sigmoid here; BCEWithLogitsLoss expects raw logits
        self.output_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        # squeeze removes the trailing dim-1 to match (batch,) target shape
        return self.output_head(x).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_resfnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_split: float = 0.1,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 3e-4,
    patience: int = 15,
    dropout: float = 0.3,
    weight_decay: float = 1e-2,
    random_state: int = 42,
) -> tuple[ResFNN, dict, StandardScaler]:
    """
    Train the ResFNN with early stopping and a learning-rate scheduler.

    Key differences from train_fnn
        - AdamW with weight_decay for L2 regularisation
        - ReduceLROnPlateau halves the LR after 5 epochs without improvement
        - BCEWithLogitsLoss + pos_weight corrects for class imbalance
        - Best weights are restored regardless of early stopping trigger

    Parameters
        X_train      : training feature DataFrame (output of get_X_y)
        y_train      : training target Series (1 = pass, 0 = run)
        val_split    : fraction used for internal validation (default 0.1)
        epochs       : maximum training epochs (default 200)
        batch_size   : mini-batch size (default 512)
        lr           : initial AdamW learning rate (default 3e-4)
        patience     : early stopping patience in epochs (default 15)
        dropout      : dropout probability for residual blocks (default 0.3)
        weight_decay : L2 regularisation strength for AdamW (default 1e-2)
        random_state : seed for reproducibility

    Returns
        model   : trained ResFNN with best validation weights restored
        history : dict with train_loss, val_loss, and lr per epoch
        scaler  : fitted StandardScaler (needed by ResFNNWrapper)
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

    # pos_weight = n_negative / n_positive  — upweights the minority class in loss
    y_arr      = y_train.values
    n_pos      = y_arr.sum()
    n_neg      = len(y_arr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # model, loss, optimiser, scheduler
    n_features = X_train.shape[1]
    model      = ResFNN(n_features=n_features, dropout=dropout)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,        # halve LR on plateau
        patience=5,        # wait 5 epochs before reducing
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
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- validation pass ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
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
            print(f"\n[train_resfnn] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_weights)
    print(f"[train_resfnn] Training complete. Best val_loss: {best_val_loss:.4f}")

    return model, history, scaler


# ---------------------------------------------------------------------------
# Wrapper  (sklearn-compatible interface for evaluate_model / plot functions)
# ---------------------------------------------------------------------------

class ResFNNWrapper:
    """
    Sklearn-style wrapper around a trained ResFNN.

    The model outputs raw logits; this wrapper applies sigmoid to convert
    them to probabilities and a threshold to produce hard predictions.

    Usage
        wrapper = ResFNNWrapper(model, scaler)
        metrics = evaluate_model(wrapper, X_test, y_test, "ResFNN", "final")
    """

    def __init__(self, model: ResFNN, scaler: StandardScaler, threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
        with torch.no_grad():
            # sigmoid converts logits → probabilities
            proba_pass = torch.sigmoid(self.model(X_tensor)).numpy()
        proba_run = 1 - proba_pass
        return np.column_stack([proba_run, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
