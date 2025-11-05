# model.py
from stock_model import AdvancedStockPredictor
import traceback

def model_predict(symbol: str, horizon: int = 1, features=None):
    """
    This function acts as a bridge between the FastAPI backend and the ML script.
    It takes a stock symbol, runs your full ML pipeline from stock_model.py,
    and returns a summarized prediction output.
    """
    try:
        # Ensure correct Yahoo Finance symbol format
        if not symbol.endswith((".NS", ".BO")) and not symbol.startswith("^"):
            symbol = symbol + ".NS"

        # Initialize your predictor class
        predictor = AdvancedStockPredictor(symbol, period="2y")

        # Fetch and process data
        if not predictor.fetch_data():
            return {"error": f"Unable to fetch data for {symbol}"}

        predictor.calculate_advanced_features()
        predictor.train_model()

        # Get model’s current prediction
        prediction, probability = predictor.get_current_prediction()
        signals = predictor.analyze_technical_signals()

        # Calculate confidence score
        confidence = max(probability) * 100

        # Combine ML and technical recommendation logic
        if prediction == 1:
            direction = "UP"
            ml_score = confidence
        else:
            direction = "DOWN"
            ml_score = 100 - confidence

        # Count buy/sell indicators
        buy_count = sum(1 for s in signals.values() if "BUY" in s["status"] or "BULLISH" in s["status"])
        sell_count = sum(1 for s in signals.values() if "SELL" in s["status"] or "BEARISH" in s["status"])

        technical_score = (buy_count / (buy_count + sell_count)) * 100 if (buy_count + sell_count) > 0 else 50
        final_score = (ml_score * 0.6) + (technical_score * 0.4)

        if final_score >= 65:
            recommendation = "STRONG BUY"
        elif final_score >= 50:
            recommendation = "MODERATE BUY"
        elif final_score >= 35:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID/SELL"

        return {
            "symbol": symbol,
            "prediction": direction,
            "confidence": round(confidence, 2),
            "final_score": round(final_score, 2),
            "recommendation": recommendation,
            "model": predictor.model_name,
            "train_accuracy": round(predictor.train_accuracy * 100, 2),
            "test_accuracy": round(predictor.test_accuracy * 100, 2)
        }

    except Exception as e:
        print("Error during prediction:", traceback.format_exc())
        return {"error": str(e)}
