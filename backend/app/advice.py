"""
advice.py
Combines the LSTM price prediction + sentiment score into a recommendation,
and layers age-based financial advice on top (the personalization layer
from your original school project).
"""


def get_age_based_advice(age: int) -> str:
    if age < 25:
        return (
            "You're early in your career. You can typically afford higher risk — "
            "consider growth stocks and a longer holding horizon."
        )
    elif age < 40:
        return (
            "You're in a wealth-building phase. A balanced mix of growth stocks "
            "and some stable blue-chip holdings is usually reasonable."
        )
    elif age < 55:
        return (
            "You're approaching peak earning years. Consider shifting some weight "
            "toward stable, dividend-paying stocks to reduce volatility."
        )
    else:
        return (
            "Closer to or in retirement, capital preservation matters more. "
            "Favor low-volatility, established stocks over speculative growth picks."
        )


def get_recommendation(current_price: float, predicted_price: float, sentiment_label: str) -> dict:
    """
    Fuses the LSTM price trend with news sentiment into a buy/sell/hold call.
    This is intentionally simple and transparent — good for explaining in an
    interview: 'here's exactly why the model said what it said.'
    """
    pct_change = ((predicted_price - current_price) / current_price) * 100

    # Base signal from price prediction
    if pct_change > 1.5:
        price_signal = "buy"
    elif pct_change < -1.5:
        price_signal = "sell"
    else:
        price_signal = "hold"

    # Sentiment can reinforce or soften the signal
    if price_signal == "buy" and sentiment_label == "negative":
        final = "hold"
        reason = "Price model suggests upside, but recent news sentiment is negative — recommend caution."
    elif price_signal == "sell" and sentiment_label == "positive":
        final = "hold"
        reason = "Price model suggests downside, but recent news sentiment is positive — signals are mixed."
    else:
        final = price_signal
        reason = f"Price model predicts a {pct_change:.2f}% move and news sentiment is {sentiment_label}, both pointing the same direction."

    return {
        "recommendation": final,
        "predicted_change_pct": round(pct_change, 2),
        "sentiment_label": sentiment_label,
        "reason": reason,
    }
