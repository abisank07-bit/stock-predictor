"""
search.py
Looks up ticker symbols by company name or partial symbol, using Yahoo
Finance's public autocomplete endpoint (the same one their own search box
uses). Powers the "type MS, see MSFT" autocomplete in the frontend.
"""

import requests
import re

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

HEADERS = {
    # Yahoo's endpoint blocks requests with no user-agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def search_tickers(query: str, limit: int = 8):
    """
    Returns a list of {symbol, name, exchange} matches for a partial
    company name or symbol, e.g. "ms" -> MSFT (Microsoft Corporation), etc.
    """

    if not query or len(query.strip()) < 1:
        return []

    params = {
        "q": query,
        "quotesCount": limit,
        "newsCount": 0,
    }

    try:
        resp = requests.get(
            YAHOO_SEARCH_URL,
            params=params,
            headers=HEADERS,
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()

    except (requests.RequestException, ValueError):
        return []

    # Only actual equities and ETFs work with the prediction flow.
    ALLOWED_TYPES = {"EQUITY", "ETF"}

    # Only allow common US exchanges.
    ALLOWED_EXCHANGES = {
        "NMS",  # Nasdaq
        "NYQ",  # NYSE
        "ASE",  # NYSE American
        "BTS",  # Cboe/US
    }

    results = []

    for quote in data.get("quotes", []):
        symbol = quote.get("symbol")
        name = quote.get("shortname") or quote.get("longname")
        exchange = quote.get("exchange")
        quote_type = quote.get("quoteType")

        # Only allow stocks and ETFs
        if quote_type not in ALLOWED_TYPES:
            continue

        # Only allow US exchanges
        if exchange not in ALLOWED_EXCHANGES:
            continue

        if not symbol or not name:
            continue

        # Reject foreign-market symbols such as AAPL.TO
        if "." in symbol:
            continue

        # Reject option-contract style symbols
        if not re.fullmatch(
            r"[A-Z]{1,5}(?:-[A-Z]{1,3})?",
            symbol
        ):
            continue

        results.append({
            "symbol": symbol,
            "name": name,
            "exchange": exchange
        })

        if len(results) >= limit:
            break

    return results
