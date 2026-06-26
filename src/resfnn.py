# src/resfnn.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config import RESFNN_DEFAULTS, RESFNN_PARAMS


# ---------------------------------------------------------------------------
# Dataset
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

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )

        self.skip = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ResFNN(nn.Module):
    """
    Residual Feedforward Neural Network for binary play-call classification.

    Architecture (with default proj_dim=128)
        Input projection : Linear(n_features → proj_dim)   → BN → ReLU → Dropout
        ResidualBlock    : proj_dim     → proj_dim // 2    (projects skip)
        ResidualBlock    : proj_dim//2  → proj_dim // 4    (projects skip)
        Output head      : Linear(proj_dim // 4 → 1)

    All dimension and dropout values are sourced from config.RESFNN_DEFAULTS
    / RESFNN_PARAMS so there are no magic numbers in this file.
    """

    def __init__(self, n_features: int, proj_dim: int, dropout: float):
        super().__init__()

        d1 = proj_dim
        d2 = proj_dim // 2
        d3 = proj_dim // 4

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d1),
            nn.BatchNorm1d(d1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_block1 = ResidualBlock(d1, d2, dropout=dropout)
        self.res_block2 = ResidualBlock(d2, d3, dropout=max(dropout - 0.1, 0.1))

        self.output_head = nn.Linear(d3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.output_head(x).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_resfnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "comprehensive",
    **overrides,
) -> tuple["ResFNN", dict, StandardScaler]:
    """
    Train the ResFNN with early stopping and a learning-rate scheduler.

    Parameters are resolved in three layers (last wins):
        1. RESFNN_DEFAULTS       — global defaults from config
        2. RESFNN_PARAMS[feature_set] — per-feature-set overrides from config
        3. **overrides           — caller-supplied keyword overrides

    Parameters
        X_train      : training feature DataFrame
        y_train      : training target Series (1 = pass, 0 = run)
        feature_set  : one of "mini" / "comprehensive" / "maxi"
        **overrides  : any key from RESFNN_DEFAULTS to override at call-time

    Returns
        model   : trained ResFNN with best validation weights restored
        history : dict with train_loss, val_loss, lr per epoch
        scaler  : fitted StandardScaler (needed by ResFNNWrapper)
    """
    from config import RESFNN_PARAMS

    # resolve final config: defaults → per-set → caller overrides
    cfg = {**RESFNN_DEFAULTS, **RESFNN_PARAMS.get(feature_set, {}), **overrides}

    # unpack
    proj_dim      = cfg["proj_dim"]
    dropout       = cfg["dropout"]
    epochs        = cfg["epochs"]
    batch_size    = cfg["batch_size"]
    val_split     = cfg["val_split"]
    lr            = cfg["lr"]
    patience      = cfg["patience"]
    weight_decay  = cfg["weight_decay"]
    random_state  = cfg["random_state"]
    sched_factor  = cfg["sched_factor"]
    sched_patience= cfg["sched_patience"]
    min_lr        = cfg["min_lr"]

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # scale features
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # dataset + split
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

    # pos_weight = n_negative / n_positive
    y_arr      = y_train.values
    n_pos      = y_arr.sum()
    n_neg      = len(y_arr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # model, loss, optimiser, scheduler
    model     = ResFNN(n_features=X_train.shape[1], proj_dim=proj_dim, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_factor,
        patience=sched_patience,
        min_lr=min_lr,
    )

    history           = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):

        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_losses.append(criterion(model(X_batch), y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

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
# Wrapper
# ---------------------------------------------------------------------------

class ResFNNWrapper:
    """
    Sklearn-style wrapper around a trained ResFNN.

    Usage
        wrapper = ResFNNWrapper(model, scaler)
        metrics = evaluate_model(wrapper, X_test, y_test, "ResFNN", "comprehensive")
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
            proba_pass = torch.sigmoid(self.model(X_tensor)).numpy()
        return np.column_stack([1 - proba_pass, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)