import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from config import FNN_DEFAULTS, FNN_PARAMS


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
# Model architecture
# ---------------------------------------------------------------------------

class FNN(nn.Module):
    """
    Feedforward Neural Network for binary classification.

    Architecture is driven by `hidden_dims` from config:
        Linear(n_features → hidden_dims[0]) → ReLU → Dropout
        ...
        Linear(hidden_dims[-1] → 1)         → Sigmoid

    Parameters
        n_features  : number of input features
        hidden_dims : list of hidden layer widths  (from FNN_DEFAULTS)
        dropout     : dropout probability after each hidden layer
    """

    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _resolve_params(feature_set: str | None) -> dict:
    """Merge FNN_DEFAULTS with feature-set-specific overrides."""
    params = dict(FNN_DEFAULTS)
    if feature_set is not None:
        params.update(FNN_PARAMS.get(feature_set, {}))
    return params


def train_fnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str | None = None,   # "mini" | "comprehensive" | "maxi"
    **overrides,
) -> tuple["FNN", dict, StandardScaler]:
    """
    Train the FNN with early stopping on a validation split.

    Hyper-parameters are resolved in order:
        FNN_DEFAULTS  →  FNN_PARAMS[feature_set]  →  **overrides

    Parameters
        X_train     : training feature df (output of get_X_y)
        y_train     : training target Series
        feature_set : one of "mini" / "comprehensive" / "maxi" (optional)
        **overrides : any FNN_DEFAULTS key to override ad-hoc
    """
    p = _resolve_params(feature_set)
    p.update(overrides)

    # --- reproducibility ---
    torch.manual_seed(p["random_state"])
    np.random.seed(p["random_state"])

    # --- scaling ---
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # --- datasets ---
    full_dataset = PlayCallDataset(X_train, y_train)
    val_size     = int(len(full_dataset) * p["val_split"])
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(p["random_state"]),
    )
    train_loader = DataLoader(train_dataset, batch_size=p["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=p["batch_size"], shuffle=False)

    # --- model / loss / optimiser ---
    model     = FNN(n_features=X_train.shape[1], hidden_dims=p["hidden_dims"], dropout=p["dropout"])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])

    # --- training loop ---
    history           = {"train_loss": [], "val_loss": []}
    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(1, p["epochs"] + 1):

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
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve == p["patience"]:
            print(
                f"  Epoch {epoch:>3} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"no_improve: {epochs_no_improve}/{p['patience']}"
            )

        if epochs_no_improve >= p["patience"]:
            print(f"\n[train_fnn] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_weights)
    print(f"[train_fnn] Training complete. Best val_loss: {best_val_loss:.4f}")
    return model, history, scaler


# ---------------------------------------------------------------------------
# Sklearn-style wrapper
# ---------------------------------------------------------------------------

class FNNWrapper:
    """
    Drop-in sklearn-style wrapper around a trained FNN.

    Usage
        wrapper = FNNWrapper(model, scaler)
        metrics = evaluate_model(wrapper, X_test, y_test, "FNN", "mini")
    """

    def __init__(self, model: FNN, scaler: StandardScaler, threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
        with torch.no_grad():
            proba_pass = self.model(X_tensor).numpy()
        return np.column_stack([1 - proba_pass, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)