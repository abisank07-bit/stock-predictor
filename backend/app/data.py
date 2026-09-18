"""
data.py
Fetches historical stock data using yfinance and prepares it for the model.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.

    ticker: e.g. "AAPL", "INFY.NS", "ICICIBANK.NS"
    period: how far back e.g. "1y", "2y", "5y"
    interval: "1d", "1wk" etc.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df


def get_last_n_days(ticker: str, n: int = 30) -> pd.DataFrame:
    """Return the last n days of closing prices — used for the line graph."""
    df = fetch_stock_data(ticker, period="3mo")
    return df.tail(n)


def prepare_sequences(df: pd.DataFrame, lookback: int = 60):
    """
    Convert closing prices into (X, y) sequences for LSTM training.
    lookback: how many past days the model looks at to predict the next day.
    """
    closes = df["Close"].values.reshape(-1, 1)

    # Normalize between 0 and 1 (LSTMs train much better on scaled data)
    min_val, max_val = closes.min(), closes.max()
    scaled = (closes - min_val) / (max_val - min_val)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # reshape for LSTM: [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, min_val, max_val
