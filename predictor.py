# predictor.py
def predict(symbol, horizon):
    # Example placeholder
    # Replace this with your actual ML model logic later
    predicted_price = 450 + (horizon * 2.5)
    return {
        "symbol": symbol,
        "predicted_price": predicted_price,
        "model_version": "v1"
    }
