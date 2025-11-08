import os
import math
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
INPUT_FEATURES = None
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------
# Utilities
# ---------------------------
def download_stock_data(ticker="^NSEI", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or getattr(df, "empty", True):
        raise ValueError(f"No data found for ticker {ticker}")
    df = df.rename(columns={"Adj Close": "Adj_Close"}).reset_index().set_index("Date")
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change().fillna(0)
    df["log_return"] = pd.Series(np.log1p(df["return"])).fillna(0)
    for w in (5, 10, 20, 50):
        df[f"ma_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    for w in (10, 20):
        df[f"std_{w}"] = df["Close"].rolling(window=w, min_periods=1).std().fillna(0)
    df["vol_change"] = df["Volume"].pct_change().fillna(0)
    df["price_vol_ratio"] = df["Close"] / (df["Volume"].replace(0, 1))
    df["future_return_1d"] = df["Close"].pct_change().shift(-1)
    # ✅ Fix infinity/NaN issue
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def prepare_sequences(df, feature_columns, seq_length=60):
    data = df[feature_columns].values
    fut = df["future_return_1d"].values
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        future = fut[i+seq_length-1]
        if np.isnan(future): continue
        if future > 0.002: label = 0  # BUY
        elif future < -0.002: label = 2  # SELL
        else: label = 1  # HOLD
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

# ---------------------------
# Model
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pos_embedding", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = getattr(self, "pos_embedding", None)
        if pe is None:
            raise RuntimeError("Positional encoding buffer not initialized.")
        if seq_len > pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {pe.size(1)}")
        return x + pe[:, :seq_len, :]

class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, num_classes=3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 128, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.fc_out(self.pool(x).squeeze(-1))

# ---------------------------
# Train or load
# ---------------------------
def train_and_save_model(ticker="^NSEI", epochs=3):
    print("➡️ Training model...")
    df = create_advanced_features(download_stock_data(ticker))
    feat_cols = [c for c in df.columns if c != "future_return_1d"]
    scaler = RobustScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    seqs, labels = prepare_sequences(df, feat_cols)
    model = ImprovedTransformerModel(len(feat_cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(seqs))
        X, y = torch.FloatTensor(seqs[idx]), torch.LongTensor(labels[idx])
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        print(f"Epoch {ep+1}: loss={loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model trained & saved.")
    return model, scaler, feat_cols

def ensure_model_and_scaler(ticker="^NSEI"):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            df = create_advanced_features(download_stock_data(ticker))
            feat_cols = [c for c in df.columns if c != "future_return_1d"]
            model = ImprovedTransformerModel(len(feat_cols))
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            print("✅ Loaded existing model and scaler.")
            return model, scaler, feat_cols
        except Exception as e:
            print(f"⚠️ Error loading model: {e}, retraining...")
    return train_and_save_model(ticker)

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="Indian Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://indianstockpredictor.netlify.app"],  # replace with Netlify URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model, _scaler, _features = None, None, None

@app.on_event("startup")
def startup_event():
    global _model, _scaler, _features
    print("🔍 Checking for model files...")
    _model, _scaler, _features = ensure_model_and_scaler("^NSEI")
    print("✅ Model ready for predictions.")

@app.get("/")
def root():
    return {"message": "✅ Indian Stock Predictor API is live!"}

@app.get("/predict")
def predict(ticker: str = "^NSEI"):
    try:
        df = create_advanced_features(download_stock_data(ticker, "1y"))
        if _features is None:
            return {"error": "Model features are not initialized."}
        if _scaler is None:
            return {"error": "Scaler is not initialized."}
        for col in _features:
            if col not in df.columns:
                df[col] = 0.0
        df[_features] = _scaler.transform(df[_features])
        seqs, _ = prepare_sequences(df, _features)
        if len(seqs) == 0:
            return {"error": "Insufficient data for prediction."}
        last = torch.FloatTensor(seqs[-1:]).to(DEVICE)
        if _model is None:
            return {"error": "Model not initialized."}
        with torch.no_grad():
            probs = torch.softmax(_model(last), dim=1).cpu().numpy()[0]
        signal = ["BUY", "HOLD", "SELL"][int(np.argmax(probs))]
        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": {
                "buy": round(float(probs[0]), 4),
                "hold": round(float(probs[1]), 4),
                "sell": round(float(probs[2]), 4),
            },
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# ✅ FIXED Render-Compatible Server Start
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
