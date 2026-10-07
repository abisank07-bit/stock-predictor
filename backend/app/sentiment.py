"""
sentiment.py
Fetches recent news headlines for a ticker and scores sentiment with VADER.
(VADER is lightweight and needs no big model download — good for a portfolio
project. Swap in FinBERT later if you want finance-tuned sentiment.)
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

analyzer = SentimentIntensityAnalyzer()


def get_news_headlines(ticker: str, limit: int = 8):
    """Pull recent news headlines for a ticker via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        headlines = [item.get("title", "") for item in news[:limit] if item.get("title")]
        return headlines
    except Exception:
        # Yahoo Finance's news endpoint can rate-limit or fail intermittently,
        # especially on shared/cloud IPs. Fall back to no headlines (which
        # scores as neutral) instead of crashing the whole /predict request.
        return []


def score_sentiment(headlines: list[str]):
    """
    Score each headline with VADER and return an aggregate.
    compound score ranges from -1 (very negative) to +1 (very positive).
    """
    if not headlines:
        return {"headlines": [], "average_compound": 0.0, "label": "neutral"}

    scored = []
    total = 0
    for h in headlines:
        vs = analyzer.polarity_scores(h)
        scored.append({"headline": h, "compound": vs["compound"]})
        total += vs["compound"]

    avg = total / len(headlines)

    if avg >= 0.2:
        label = "positive"
    elif avg <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {"headlines": scored, "average_compound": round(avg, 3), "label": label}


def get_sentiment_for_ticker(ticker: str):
    headlines = get_news_headlines(ticker)
    return score_sentiment(headlines)
