# app.py
import os
import math
import time
from typing import List

import uvicorn
import yfinance as yf
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

# ---------------------------
# Config / Hyperparameters
# ---------------------------
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler.pkl"
SEQ_LENGTH = 60
INPUT_FEATURES = None  # will be set after features are created
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Utilities: data download + features
# ---------------------------
def download_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for the ticker using yfinance.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # yf.download may return None or an empty DataFrame; check safely before using .empty
    if df is None or getattr(df, "empty", True):
        raise ValueError(f"No data found for ticker {ticker}")
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df = df.reset_index().set_index("Date")
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few basic technical features:
    - daily returns
    - rolling means
    - rolling std
    - volume change
    Keep it simple and deterministic.
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change().fillna(0)
    # ensure a pandas Series so .fillna is available to the type-checker
    df["log_return"] = pd.Series(np.log1p(df["return"]), index=df.index).fillna(0)

    # moving averages
    for w in (5, 10, 20, 50):
        df[f"ma_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()

    # rolling std
    for w in (10, 20):
        df[f"std_{w}"] = df["Close"].rolling(window=w, min_periods=1).std().fillna(0)

    # volume related
    df["vol_change"] = df["Volume"].pct_change().fillna(0)
    df["price_vol_ratio"] = df["Close"] / (df["Volume"].replace(0, 1))

    # forward pct change for labeling (next day)
    df["future_return_1d"] = df["Close"].pct_change().shift(-1)
    df = df.dropna(how="all")
    return df

def prepare_sequences(df: pd.DataFrame, feature_columns: List[str], seq_length: int = 60):
    """
    Create sequences and labels.
    Labels:
    - BUY  -> future_return_1d > threshold_pos
    - HOLD -> -threshold_pos <= future_return_1d <= threshold_pos
    - SELL -> future_return_1d < -threshold_pos
    """
    data = df[feature_columns].values
    fut = df["future_return_1d"].values
    sequences = []
    labels = []
    threshold = 0.002  # 0.2% threshold for classification, adjust as needed

    for i in range(len(data) - seq_length):
        seq = data[i: i + seq_length]
        future = fut[i + seq_length - 1]  # using the last element's future_return_1d
        if np.isnan(future):
            continue
        if future > threshold:
            label = 0  # BUY
        elif future < -threshold:
            label = 2  # SELL
        else:
            label = 1  # HOLD
        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return sequences, labels

# ---------------------------
# PyTorch Dataset
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, pad last column with zeros for cosine
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # use getattr to fetch the buffer as a tensor (helps static type-checkers)
        pe = getattr(self, "pe")
        x = x + pe[:, :seq_len]
        return x

# ---------------------------
# Transformer Model
# ---------------------------
class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # to pool over sequence dimension after permute
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_fc(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # Pool over seq_len
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)

        
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        out = self.fc_out(x)  # (batch, num_classes)
        return out

# ---------------------------
# Training routine
# ---------------------------
def train_and_save_model(ticker="^NSEI", epochs=10, batch_size=32, lr=1e-3):
    """
    Train a model using data for `ticker` (default ^NSEI) and save model & scaler.
    If model files already exist, training is skipped by caller.
    """
    print("➡️ Downloading data for training...")
    df = download_stock_data(ticker, period="3y", interval="1d")
    df = create_advanced_features(df)

    # Choose feature columns (keep core OHLCV and created features)
    exclude_cols = ['future_return_1d']
    feature_columns = [c for c in df.columns if c not in exclude_cols]
    # Provide stable ordering: place Open..Close..Volume first if present
    base = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj_Close"] if c in feature_columns]
    others = [c for c in feature_columns if c not in base]
    feature_columns = base + others

    global INPUT_FEATURES
    INPUT_FEATURES = len(feature_columns)

    # Fit scaler on full dataset features
    scaler = RobustScaler()
    features_mat = df[feature_columns].values
    scaler.fit(features_mat)
    X_scaled = scaler.transform(features_mat)
    df_scaled = df.copy()
    df_scaled[feature_columns] = X_scaled

    # sequences and labels
    sequences, labels = prepare_sequences(df_scaled, feature_columns, SEQ_LENGTH)
    if len(sequences) == 0:
        raise RuntimeError("Insufficient data after processing to train. Try longer period or smaller SEQ_LENGTH.")

    # split train/val
    split = int(0.85 * len(sequences))
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split], indices[split:]

    train_seqs, val_seqs = sequences[train_idx], sequences[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    train_ds = SeqDataset(train_seqs, train_labels)
    val_ds = SeqDataset(val_seqs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = ImprovedTransformerModel(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("➡️ Starting training...")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            print(f"✅ Saved best model and scaler at epoch {epoch} (val_loss: {val_loss:.4f})")

    print("➡️ Training complete.")

# ---------------------------
# Helper: load model & scaler (or train if absent)
# ---------------------------
def ensure_model_and_scaler(ticker="^NSEI"):
    global INPUT_FEATURES
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            # We need feature count — try to infer using a quick data download & feature creation
            df = download_stock_data(ticker, period="1y", interval="1d")
            df = create_advanced_features(df)
            feature_columns = [c for c in df.columns if c not in ['future_return_1d']]
            base = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj_Close"] if c in feature_columns]
            feature_columns = base + [c for c in feature_columns if c not in base]
            INPUT_FEATURES = len(feature_columns)

            scaler = joblib.load(SCALER_PATH)
            model = ImprovedTransformerModel(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            print("✅ Model and scaler loaded successfully.")
            return model, scaler, feature_columns
        except Exception as e:
            print(f"⚠️ Failed to load model/scaler properly: {e}. Will retrain.")
    # Otherwise train
    train_and_save_model(ticker=ticker, epochs=10, batch_size=32, lr=1e-3)
    # Load after training
    scaler = joblib.load(SCALER_PATH)
    # infer features again
    df = download_stock_data(ticker, period="1y", interval="1d")
    df = create_advanced_features(df)
    feature_columns = [c for c in df.columns if c not in ['future_return_1d']]
    base = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj_Close"] if c in feature_columns]
    feature_columns = base + [c for c in feature_columns if c not in base]
    INPUT_FEATURES = len(feature_columns)
    model = ImprovedTransformerModel(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("✅ Model and scaler ready after training.")
    return model, scaler, feature_columns

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Indian Stock Predictor API (single-file)",
    description="Transformer-based backend for stock trend prediction (single-file: training + API)",
    version="1.0"
)

# CORS - allow all for development; change in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://indianstockpredictor.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load on first request
_model = None
_scaler = None
_feature_columns = None

@app.on_event("startup")
def load_on_startup():
    global _model, _scaler, _feature_columns
    try:
        _model, _scaler, _feature_columns = ensure_model_and_scaler(ticker="^NSEI")
    except Exception as e:
        print(f"⚠️ Startup: unable to prepare model/scaler: {e}")

@app.get("/")
def root():
    return {"message": "✅ Indian Stock Predictor API (single-file) is running successfully!"}

@app.get("/predict")
def predict(ticker: str = Query(default="^NSEI")):
    """
    Download recent data for `ticker`, create features, scale using saved scaler,
    build last sequence and return model's classification (BUY/HOLD/SELL) + confidences.
    """
    try:
        global _model, _scaler, _feature_columns
        if _model is None or _scaler is None or _feature_columns is None:
            _model, _scaler, _feature_columns = ensure_model_and_scaler(ticker)

        df = download_stock_data(ticker, period="1y", interval="1d")
        df = create_advanced_features(df)

        feature_columns = [c for c in df.columns if c in _feature_columns]
        if len(feature_columns) != len(_feature_columns):
            # ensure consistent ordering & columns: try to keep the original list and use zeros for missing
            ordered_features = _feature_columns
            for col in ordered_features:
                if col not in df.columns:
                    df[col] = 0.0
            feature_columns = ordered_features

        # Scale
        df_scaled = df.copy()
        df_scaled[feature_columns] = _scaler.transform(df[feature_columns])

        sequences, labels = prepare_sequences(df_scaled, feature_columns, SEQ_LENGTH)
        if len(sequences) == 0:
            return {"error": "Insufficient data for prediction. Try longer period or different ticker."}

        last_seq = torch.FloatTensor(sequences[-1:])  # shape (1, seq_len, input_dim)
        _model.to(DEVICE)
        _model.eval()
        with torch.no_grad():
            out = _model(last_seq.to(DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            signal = ["BUY", "HOLD", "SELL"][pred_class]
            return {
                "ticker": ticker,
                "signal": signal,
                "confidence": {
                    "buy": round(float(probs[0]), 4),
                    "hold": round(float(probs[1]), 4),
                    "sell": round(float(probs[2]), 4),
                }
            }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    # Allow direct start: will run startup event then start uvicorn
    # We call ensure_model_and_scaler() here to show progress in terminal before Uvicorn boots,
    # so user sees training logs immediately when files are absent.
    try:
        print("Starting single-file app. Preparing model and scaler (training if needed)...")
        ensure_model_and_scaler(ticker="^NSEI")
    except Exception as e:
        print(f"⚠️ Pre-start failure: {e}")
    print("Launching FastAPI (uvicorn)...")
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
