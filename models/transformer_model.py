"""
models/transformer_model.py
─────────────────────────────────────────────────────────────
Temporal Fusion Transformer (TFT) built in pure PyTorch.
State-of-the-art time-series forecasting architecture.
Much more accurate than vanilla LSTM on financial data.

Architecture:
  Input → Variable Selection → LSTM Encoder → 
  Multi-Head Self-Attention → Gated Residual → Output

Usage:
  from models.transformer_model import train_transformer, predict_transformer
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models/saved"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"PyTorch device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Sliding window dataset for time-series sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, features)
        y_val  = self.y[idx + self.seq_len]                # scalar target
        return x_seq, y_val


# ── Gated Residual Network ─────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """Core building block of TFT — gated skip connection with ELU."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(hidden_dim, output_dim)
        self.skip     = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout  = nn.Dropout(dropout)
        self.elu      = nn.ELU()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        h  = self.elu(self.fc1(x))
        h  = self.dropout(h)
        v  = self.fc2(h)
        g  = self.sigmoid(self.gate(h))
        out = g * v + (1 - g) * self.skip(x)
        return self.layer_norm(out)


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Temporal Fusion Transformer ────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for stock price forecasting.

    Args:
        n_features:  Number of input features per timestep
        d_model:     Embedding dimension (default 64)
        n_heads:     Attention heads (default 4)
        n_layers:    Transformer layers (default 2)
        seq_len:     Sequence length
        dropout:     Dropout rate
        task:        'regression' | 'classification'
    """
    def __init__(
        self,
        n_features: int,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        seq_len:    int   = 60,
        dropout:    float = 0.1,
        task:       str   = "regression",
    ):
        super().__init__()
        self.d_model  = d_model
        self.seq_len  = seq_len
        self.task     = task

        # ── Input projection ────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Variable selection (GRN per feature group) ──────
        self.var_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # ── Positional encoding ──────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # ── LSTM encoder (local temporal patterns) ──────────
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
            bidirectional = True,
        )
        self.lstm_proj = nn.Linear(d_model * 2, d_model)

        # ── Transformer encoder (global attention) ──────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = d_model * 4,
            dropout     = dropout,
            batch_first = True,
            norm_first  = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Gated skip from LSTM to Transformer output ──────
        self.gate_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # ── Output head ──────────────────────────────────────
        if task == "regression":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch,) predictions
        """
        # Project input
        h = self.input_proj(x)                         # (B, T, d)
        h = self.var_grn(h)                            # variable selection
        h = self.pos_enc(h)                            # positional encoding

        # LSTM local patterns
        lstm_out, _ = self.lstm(h)                     # (B, T, 2*d)
        lstm_out = self.lstm_proj(lstm_out)            # (B, T, d)

        # Transformer global attention
        attn_out = self.transformer(lstm_out)          # (B, T, d)

        # Gated fusion
        fused = self.gate_grn(attn_out + lstm_out)    # (B, T, d)

        # Use last timestep for prediction
        out = self.head(fused[:, -1, :])               # (B, 1)
        return out.squeeze(-1)                         # (B,)


# ── Training ──────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    ticker:  str   = "AAPL",
    task:    str   = "regression",
    seq_len: int   = 60,
    d_model: int   = 64,
    n_heads: int   = 4,
    n_layers:int   = 2,
    epochs:  int   = 100,
    batch_size:int = 32,
    lr:      float = 1e-3,
    dropout: float = 0.1,
) -> tuple:
    """
    Train TFT model and return (model, train_losses, val_losses).
    """
    n_features = X_train.shape[1]
    model = TemporalFusionTransformer(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, seq_len=seq_len, dropout=dropout, task=task
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TFT model: {total_params:,} trainable parameters")

    train_ds = StockSequenceDataset(X_train, y_train, seq_len)
    val_ds   = StockSequenceDataset(
        np.vstack([X_train[-seq_len:], X_val]),
        np.concatenate([y_train[-seq_len:], y_val]),
        seq_len
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    if task == "regression":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)
    es = EarlyStopping(patience=15)

    train_losses, val_losses = [], []
    best_state = None; best_val = float("inf")

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        # ── Validate ─────────────────────────────────────────
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item()

        t_loss /= max(len(train_loader), 1)
        v_loss /= max(len(val_loader),   1)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step()

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}  train={t_loss:.5f}  val={v_loss:.5f}  lr={scheduler.get_last_lr()[0]:.6f}")

        es(v_loss)
        if es.stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"Best val loss: {best_val:.5f}")
    return model, train_losses, val_losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_transformer(
    model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    seq_len: int = 60,
    task:    str = "regression",
) -> dict:
    model.eval()
    ds     = StockSequenceDataset(X_test, y_test, seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds  = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb.to(DEVICE)).cpu().numpy())
    preds  = np.array(preds)
    actual = y_test[seq_len:]

    if task == "regression":
        mae  = float(np.mean(np.abs(actual - preds)))
        rmse = float(np.sqrt(np.mean((actual - preds)**2)))
        r2   = float(1 - np.sum((actual - preds)**2) / (np.sum((actual - actual.mean())**2) + 1e-9))
        da   = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(preds))))
        logger.info(f"TFT Regression → MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  DirAcc={da:.1%}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "DirAcc": da, "predictions": preds}
    else:
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_bin = (preds > 0.5).astype(int)
        acc   = accuracy_score(actual, y_bin)
        auc   = roc_auc_score(actual, preds)
        logger.info(f"TFT Classification → Acc={acc:.1%}  AUC={auc:.3f}")
        return {"Accuracy": acc, "AUC": auc, "predictions": preds}


# ── Predict single step ───────────────────────────────────────────────────────

def predict_transformer(
    model,
    X_recent: np.ndarray,
    seq_len:  int = 60,
) -> float:
    """Predict next value from the most recent `seq_len` rows of features."""
    model.eval()
    if len(X_recent) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(X_recent)}")
    x = torch.tensor(X_recent[-seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x).item()
    return pred


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_transformer(model, ticker: str, task: str = "regression"):
    path = MODEL_DIR / f"tft_{ticker}_{task}.pt"
    torch.save({
        "state_dict":  model.state_dict(),
        "config": {
            "n_features": model.input_proj[0].in_features,
            "d_model":    model.d_model,
            "seq_len":    model.seq_len,
            "task":       model.task,
        }
    }, path)
    logger.info(f"TFT saved → {path}")


def load_transformer(ticker: str, task: str = "regression") -> TemporalFusionTransformer:
    path = MODEL_DIR / f"tft_{ticker}_{task}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No saved TFT at {path}. Train first.")
    ckpt   = torch.load(path, map_location=DEVICE)
    cfg    = ckpt["config"]
    model  = TemporalFusionTransformer(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"TFT loaded ← {path}")
    return model


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Smoke-testing TFT with random data…")
    N, F, SEQ = 800, 50, 60
    X = np.random.randn(N, F).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    tr = int(0.75*N); vl = int(0.85*N)

    model, tl, vl2 = train_transformer(
        X[:tr], y[:tr], X[tr:vl], y[tr:vl],
        ticker="TEST", epochs=15, batch_size=16
    )
    metrics = evaluate_transformer(model, X[vl:], y[vl:], seq_len=SEQ)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items() if k != "predictions"})
    print(f"Single prediction: {predict_transformer(model, X[-SEQ:]):.4f}")
