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
    Wraps a feature df and target series as a PyTorch dataset.

    OPT: Tensors werden direkt auf das Ziel-Device geladen, wenn der
    gesamte Datensatz in den GPU-Speicher passt (typisch bei NFL-Daten
    mit ~150 k Zeilen). Das eliminiert den CPU→GPU-Transfer pro Batch
    vollständig und ist der größte einzelne Speed-Gewinn.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, device: torch.device | None = None):
        dev = device if device is not None else torch.device("cpu")
        self.X      = torch.tensor(X.values, dtype=torch.float32).to(dev)
        self.y      = torch.tensor(y.values, dtype=torch.float32).to(dev)
        self.on_gpu = dev.type == "cuda"

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

    OPT: Ghost Batch Normalisation (GBN) statt Standard-BN.
    GBN teilt den Batch in virtuelle Sub-Batches der Größe `virtual_batch_size`
    auf und normalisiert jeden separat. Das entkoppelt die BN-Statistiken von
    der tatsächlichen Batch-Größe und reduziert Overfitting deutlich –
    besonders hilfreich, wenn wir auf große Batches (4096) umstellen.
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
        self.fc_shared          = fc_shared
        self.virtual_batch_size = virtual_batch_size

        # Ghost BN für shared und step-spezifischen Pfad
        self.bn_shared = nn.BatchNorm1d(out_features * 2, momentum=momentum)
        self.fc_step   = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn_step   = nn.BatchNorm1d(out_features * 2, momentum=momentum)

        self.out_features = out_features
        self.scale        = (0.5 ** 0.5)

    def _ghost_bn(self, bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        """Wendet BN auf virtuelle Sub-Batches an und setzt sie wieder zusammen."""
        B = x.shape[0]
        vbs = self.virtual_batch_size

        # Wenn Batch kleiner als vbs oder kein GBN gewünscht → normales BN
        if not self.training or B <= vbs:
            return bn(x)

        # Teile in Chunks, normalisiere jeden, füge zusammen
        chunks  = x.split(vbs, dim=0)
        normed  = [bn(c) for c in chunks]
        return torch.cat(normed, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_shared = self._ghost_bn(self.bn_shared, self.fc_shared(x))
        h_step   = self._ghost_bn(self.bn_step,   self.fc_step(x))
        h        = h_shared + h_step
        return torch.nn.functional.glu(h, dim=-1) * self.scale


class TabNetStep(nn.Module):
    """
    One attentive step in TabNet.

    OPT: Dropout auf dem Attentive Transformer.
    Die Attention-Maske neigt zum Kollaps auf wenige Features (Overuse).
    Ein leichter Dropout (p=0.1) auf den Attention-Logits vor Sparsemax
    zwingt das Modell, alternative Feature-Kombinationen zu explorieren
    und verbessert die Generalisierung.
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

        self.fc_att     = nn.Linear(n_a, n_features, bias=False)
        self.bn_att     = nn.BatchNorm1d(n_features, momentum=momentum)
        # OPT: Dropout auf Attention-Logits
        self.att_drop   = nn.Dropout(p=att_dropout)

        self.glu1 = GLUBlock(n_features,   n_d + n_a, fc_shared_1, virtual_batch_size, momentum)
        self.glu2 = GLUBlock(n_d + n_a,    n_d + n_a, fc_shared_2, virtual_batch_size, momentum)

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention mask
        att_logits = self.bn_att(self.fc_att(h_prev))
        att_logits = self.att_drop(att_logits)          # OPT: Attention Dropout
        att_logits = att_logits * prior_scales
        alpha      = self._sparsemax(att_logits)

        masked_x = alpha * x

        # Feature Transformer
        h     = self.glu1(masked_x)
        h     = self.glu2(h)
        n_d   = h.shape[-1] // 2
        h_out = h[:, :n_d]
        h_a   = h[:, n_d:]

        return h_out, h_a, alpha

    @staticmethod
    def _sparsemax(z: torch.Tensor) -> torch.Tensor:
        """Sparsemax activation (Martins & Astudillo, 2016)."""
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        z_cumsum    = torch.cumsum(z_sorted, dim=-1)
        k           = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
        z_check     = 1 + k * z_sorted > z_cumsum
        k_z         = z_check.sum(dim=-1, keepdim=True)
        tau         = (z_cumsum.gather(-1, k_z - 1) - 1) / k_z.float()
        return torch.clamp(z - tau, min=0)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TabNet(nn.Module):
    """
    TabNet für binäre Play-Call-Klassifikation.

    Architektur-Änderungen gegenüber der Basisversion:
    ┌─────────────────────────────────────────────────────────────┐
    │ OPT 1 │ n_d / n_a: 32 → 64                                 │
    │        │ Mehr Kapazität pro Step. Der Val-Loss stagnierte   │
    │        │ früh — das deutet auf Underfitting der Steps,      │
    │        │ nicht auf zu viel Kapazität.                       │
    ├─────────────────────────────────────────────────────────────┤
    │ OPT 2 │ n_steps: 4 → 6                                     │
    │        │ Mehr sequentielle Attention-Steps erlauben         │
    │        │ feinere Feature-Interaktionen. XGBoost profitiert  │
    │        │ von tiefen Bäumen; TabNet braucht mehr Steps.      │
    ├─────────────────────────────────────────────────────────────┤
    │ OPT 3 │ Ghost Batch Normalisation (via GLUBlock)            │
    │        │ Stabilisiert Training mit großen Batches und       │
    │        │ reduziert Overfitting ohne zusätzliche Parameter.  │
    ├─────────────────────────────────────────────────────────────┤
    │ OPT 4 │ Finaler Dropout vor dem Output-Head (p=0.15)       │
    │        │ Klassischer Regularisierer; der Gap Train/Val      │
    │        │ in der Kurve zeigt, dass er hier nötig ist.        │
    ├─────────────────────────────────────────────────────────────┤
    │ OPT 5 │ gamma: 1.5 → 1.3                                   │
    │        │ Niedrigeres gamma = mehr Feature-Reuse erlaubt.   │
    │        │ Bei 33 Features und 6 Steps ist Reuse sinnvoll.   │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        n_features: int,
        n_d: int = 24,              # OPT 1: 32 → 64
        n_a: int = 24,              # OPT 1: 32 → 64
        n_steps: int = 3,           # OPT 2: 4 → 6
        gamma: float = 1.5,         # OPT 5: 1.5 → 1.3
        momentum: float = 0.02,
        virtual_batch_size: int = 256,
        att_dropout: float = 0.1,
        final_dropout: float = 0.15,  # OPT 4
    ):
        super().__init__()

        self.n_features = n_features
        self.n_d        = n_d
        self.n_a        = n_a
        self.n_steps    = n_steps
        self.gamma      = gamma

        self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)

        # Shared layers (weight-tied across steps)
        self.fc_shared_1 = nn.Linear(n_features,  (n_d + n_a) * 2, bias=False)
        self.fc_shared_2 = nn.Linear(n_d + n_a,   (n_d + n_a) * 2, bias=False)

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

        # OPT 4: Dropout vor Output-Head
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.output_head   = nn.Linear(n_d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_bn(x)

        prior_scales     = torch.ones(x.shape[0], self.n_features, device=x.device)
        h_a              = torch.zeros(x.shape[0], self.n_a, device=x.device)
        decision_outputs = []
        sparsity_loss    = torch.zeros(1, device=x.device)

        for step in self.steps:
            h_out, h_a, alpha = step(x, prior_scales, h_a)
            decision_outputs.append(h_out)
            prior_scales = prior_scales * (self.gamma - alpha)
            sparsity_loss += (-alpha * torch.log(alpha + 1e-15)).sum(dim=-1).mean()

        agg    = torch.stack(decision_outputs, dim=0).sum(dim=0)
        agg    = torch.relu(agg)
        agg    = self.final_dropout(agg)          # OPT 4
        logits = self.output_head(agg).squeeze(1)

        return logits, sparsity_loss / self.n_steps

    def get_feature_importances(self, x: torch.Tensor) -> np.ndarray:
        """Summe der Attention-Gewichte über alle Steps, gemittelt über den Batch."""
        self.eval()
        x            = self.initial_bn(x)
        prior_scales = torch.ones(x.shape[0], self.n_features, device=x.device)
        h_a          = torch.zeros(x.shape[0], self.n_a, device=x.device)
        total_alpha  = torch.zeros(x.shape[0], self.n_features, device=x.device)

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
    X_train, y_train,
    val_split=0.1,
    epochs=300,
    batch_size=4096,
    virtual_batch_size=512,
    lr=1e-3,
    patience=55,
    n_d=24, n_a=24,
    n_steps=3,
    gamma=1.5,
    lambda_sparse=1e-4,
    weight_decay=1e-4,
    att_dropout=0.1,            
    final_dropout=0.15,
    random_state=42,
) -> tuple["TabNet", dict, StandardScaler]:

    """
    Trainiert TabNet mit Early Stopping, LR-Scheduling, Ghost BN und Sparsity-Loss.

    Wichtigste Hyperparameter-Änderungen:
      lambda_sparse : 1e-4 → 1e-3  — stärker sparse Attention erzwingen
      weight_decay  : 1e-5 → 1e-4  — stärkeres L2 gegen Overfitting
      batch_size    : 1024 → 4096  — GPU-Auslastung + stabilere BN-Statistiken
      patience      : 20   → 25    — mehr Raum für das tiefere Modell
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tabnet] Using device: {device}")

    # Scale features — fit only on training data
    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # OPT GPU: Datensatz direkt auf GPU laden (eliminiert per-Batch Transfer)
    # → num_workers=0 nötig, da CUDA-Tensors nicht über Worker-Prozesse geteilt werden können
    preload_device = device if device.type == "cuda" else None
    full_dataset   = PlayCallDataset(X_scaled, y_train, device=preload_device)

    val_size     = int(len(full_dataset) * val_split)
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    # OPT GPU: num_workers=0 bei GPU-Preloading, sonst 2 für CPU-Betrieb
    nw = 0 if (device.type == "cuda") else 2
    pw = device.type != "cuda"   # pin_memory nur sinnvoll bei CPU-Tensors
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

    # Class imbalance: pos_weight = n_neg / n_pos
    y_arr      = y_train.values
    n_pos      = y_arr.sum()
    n_neg      = len(y_arr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    # Modell
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

    # torch.compile() deaktiviert: auf Colab Free T4 blockiert die einmalige
    # Graph-Kompilierung 10–15 Minuten und lohnt sich bei ~200 Epochs nicht.

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # OPT: AdamW bleibt, aber mit stärkerem weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OPT: CosineAnnealingWarmRestarts statt ReduceLROnPlateau
    # Cosine-Schedule ist für TabNet besser dokumentiert als Step-Reduktion:
    # der LR schwillt periodisch an und hilft, aus lokalen Minima herauszukommen.
    # T_0=50: erste Periode 50 Epochs, T_mult=1: alle Perioden gleich lang.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )

    # OPT RAM: Checkpoint auf Disk statt clone() im RAM
    best_ckpt  = os.path.join(tempfile.gettempdir(), "tabnet_best.pt")
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):

        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            if device.type != "cuda":          # bei GPU-Preloading bereits auf Device
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, sparsity = model(X_batch)
            loss = criterion(logits, y_batch) + lambda_sparse * sparsity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Cosine Schedule: pro Epoch updaten
        scheduler.step(epoch - 1)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if device.type != "cuda":
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                logits, sparsity = model(X_batch)
                loss = criterion(logits, y_batch) + lambda_sparse * sparsity
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Early Stopping + Checkpoint
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            # OPT RAM: Weights auf Disk, nicht im RAM-Dict
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == patience:
            print(
                f"  Epoch {epoch:>3} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"no_improve: {no_improve}/{patience}"
            )

        if no_improve >= patience:
            print(f"\n[train_tabnet] Early stopping at epoch {epoch}.")
            break

    # Bestes Modell laden
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[train_tabnet] Training complete. Best val_loss: {best_val:.4f}")

    return model, history, scaler


# ---------------------------------------------------------------------------
# Wrapper  (sklearn-compatible interface for evaluate_model / plot functions)
# ---------------------------------------------------------------------------

class TabNetWrapper:
    """
    Sklearn-style Wrapper um ein trainiertes TabNet.
    Interface identisch zur Vorgängerversion — kein Änderungsbedarf in evaluate_model.
    """

    def __init__(self, model: TabNet, scaler: StandardScaler, threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        device   = next(self.model.parameters()).device
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            proba_pass = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1 - proba_pass, proba_pass])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


'''


{'model': 'TabNet', 'feature_set': 'comprehensive', 'accuracy': 0.7147, 'precision': 0.7205, 'recall': 0.7147, 'f1': 0.7161, 'roc_auc': np.float64(0.7897)}
{'model': 'TabNet', 'feature_set': 'maxi', 'accuracy': 0.7121, 'precision': 0.7211, 'recall': 0.7121, 'f1': 0.7138, 'roc_auc': np.float64(0.7888)}
'''