from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import model_predict

app = FastAPI(title="Indian Stock Predictor API")

class PredictRequest(BaseModel):
    symbol: str
    horizon: int = 1

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = model_predict(req.symbol, req.horizon)
        return {
            "id": f"pred_{req.symbol}_{req.horizon}",
            "symbol": req.symbol.upper(),
            "prediction": result,
            "model_version": result.get("model_version", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
