#src/fnn.py

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
    """Wraps a feature df and target series as a PyTorch dataset"""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        # casting float32 to avoid runtime errors
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
    Two-hidden-layer Feedforward Neural Network for binary classification.

    Architecture
        Linear(n_features → 64)      → ReLU         → Dropout(p)
        Linear(64 → 32)              → ReLU         → Dropout(p)
        Linear(32 → 1)               → Sigmoid

    Parameters
        n_features  : number of input features inferred from X_train
        dropout     : dropout probability applied after each hidden layer default 0.3
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        # definiing layers seq.
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout), # helps prevent network from memorizing sprase features patterns
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(), #paired with BCEloss in training loop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # removes extra dimensions to match target shape
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_split: float = 0.1,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    patience: int = 10,
    dropout: float = 0.3,
    random_state: int = 42,
) -> tuple[FNN, dict]:
    """
    Train the FNN with early stopping on validation split

    The best model weights (lowest validation loss) are restored after
    training stops, regardless of whether early stopping was triggered

    Parameters
        X_train      : training feature df (output of get_X_y)
        y_train      : training target Series
        val_split    : fraction of training data used for validation (default 0.1)
        epochs       : maximum training epochs (default 100)
        batch_size   : mini batch size (default 512 — safe for my personal 8 GB RAM on CPU)
        lr           : Adam learning rate (default 1e-3)
        patience     : early stopping patience in epochs (default 10)
        dropout      : dropout probability (default 0.3)
        random_state : seed for reproducibility
    """
    # seeding for reporducible
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # data setsplits 
    full_dataset = PlayCallDataset(X_train, y_train)

    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # using generator with seed guarentess validation split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    # shuffle prevents from learning order based patterns
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # model, loss, optimiser 
    n_features = X_train.shape[1]
    model      = FNN(n_features=n_features, dropout=dropout)
    criterion  = nn.BCELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop 
    history         = {"train_loss": [], "val_loss": []}
    best_val_loss   = float("inf")
    best_weights    = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):

        # Train mode enables dropout active behavior
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() # clear gradient before calc. new one
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward() # calc gradients
            optimizer.step() #update weights
            train_losses.append(loss.item())

        # eval mode disables dropout layes for consistenty 
        model.eval()
        val_losses = []
        # disables gradient tracking to save memory and speed up
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()} # to copy weights instead of shraing reference
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve == patience:
            print(
                f"  Epoch {epoch:>3} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"no_improve: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print(f"\n[train_fnn] Early stopping at epoch {epoch}.")
            break

    # restore best weights to ensure the retunred model has opitmal parameters
    model.load_state_dict(best_weights)
    print(f"[train_fnn] Training complete. Best val_loss: {best_val_loss:.4f}")

    return model, history, scaler


# ---------------------------------------------------------------------------
# Predict helper  (makes FNN compatible with evaluate_model in evaluation.py)
# ---------------------------------------------------------------------------

class FNNWrapper:
    """
    short sklearn-style wrapper around a trained FNN

    Usage
        wrapper = FNNWrapper(model)
        metrics = evaluate_model(wrapper, X_test, y_test, "FNN", "final")
    """

    def __init__(self, model: FNN, scaler: StandardScaler,threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval() #ensure dropout is off during evalutation
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
        with torch.no_grad():
            proba_pass = self.model(X_tensor).numpy()
        proba_run = 1 - proba_pass
        # column stack conversts 1D arrays into 2d to match scikitlearn standard settings
        return np.column_stack([proba_run, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        # applies treshhold to convert raw probabilities to 1 or 0
        return (proba >= self.threshold).astype(int)