"""
main.py
FastAPI backend for the Stock Market Predictor.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.data import fetch_stock_data, get_last_n_days, prepare_sequences
from app.model import train_model, save_model, load_model, predict_next_price, confidence_score
from app.sentiment import get_sentiment_for_ticker
from app.advice import get_age_based_advice, get_recommendation
from app.search import search_tickers

app = FastAPI(title="Stock Market Predictor API")

# Local dev origins are always allowed. For production, set FRONTEND_URL
# (e.g. https://your-app.vercel.app) as an environment variable on your
# hosting platform (Render/Railway) — no code change needed for that.
default_origins = ["http://localhost:5173", "http://localhost:3000"]
frontend_url = os.environ.get("FRONTEND_URL")
allowed_origins = default_origins + ([frontend_url] if frontend_url else [])

app.add_middleware( CORSMiddleware, allow_origins=allowed_origins, allow_origin_regex=r"https://.*\.vercel\.app", allow_methods=["*"], allow_headers=["*"], )

LOOKBACK = 60


class PredictRequest(BaseModel):
    ticker: str
    age: int


@app.get("/")
def root():
    return {"status": "Stock Market Predictor API is running"}


@app.get("/search/{query}")
def search(query: str):
    """Ticker autocomplete — used by the frontend's search-as-you-type box."""
    return {"results": search_tickers(query)}


@app.get("/history/{ticker}")
def history(ticker: str):
    """Last 30 days of closing prices — powers the line graph."""
    try:
        df = get_last_n_days(ticker, n=30)
        return {
            "ticker": ticker,
            "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "closes": df["Close"].round(2).tolist(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/sentiment/{ticker}")
def sentiment(ticker: str):
    return get_sentiment_for_ticker(ticker)


@app.post("/train/{ticker}")
def train(ticker: str):
    """
    Train (or retrain) the LSTM for a given ticker on 2 years of data.
    In a real deployment you'd run this on a schedule, not per-request.
    """
    try:
        df = fetch_stock_data(ticker, period="2y")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if len(df) < LOOKBACK + 10:
        raise HTTPException(status_code=400, detail="Not enough historical data to train.")

    X, y, min_val, max_val = prepare_sequences(df, lookback=LOOKBACK)
    model, history = train_model(X, y, epochs=25)
    save_model(model, ticker)

    return {
        "ticker": ticker,
        "trained_on_samples": len(X),
        "final_loss": round(history[-1], 5),
        "confidence": confidence_score(history),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    ticker = req.ticker.upper()

    try:
        df = fetch_stock_data(ticker, period="2y")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = load_model(ticker)
    if model is None:
        # Auto-train on first request if no saved model exists yet
        X, y, min_val, max_val = prepare_sequences(df, lookback=LOOKBACK)
        model, hist = train_model(X, y, epochs=25)
        save_model(model, ticker)
        conf = confidence_score(hist)
    else:
        _, _, min_val, max_val = prepare_sequences(df, lookback=LOOKBACK)
        conf = None  # not retrained this call; could store/load conf separately

    # Build the most recent sequence to predict the *next* day from
    closes = df["Close"].values.reshape(-1, 1)
    scaled = (closes - min_val) / (max_val - min_val)
    last_sequence = scaled[-LOOKBACK:]

    current_price = float(df["Close"].iloc[-1])
    predicted_price = predict_next_price(model, last_sequence, min_val, max_val)

    sentiment_data = get_sentiment_for_ticker(ticker)
    recommendation = get_recommendation(current_price, predicted_price, sentiment_data["label"])
    advice = get_age_based_advice(req.age)

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "predicted_next_price": round(predicted_price, 2),
        "confidence": conf,
        "sentiment": sentiment_data,
        "recommendation": recommendation,
        "age_based_advice": advice,
    }
