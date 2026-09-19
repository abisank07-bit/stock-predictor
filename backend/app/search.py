"""
search.py
Looks up ticker symbols by company name or partial symbol, using Yahoo
Finance's public autocomplete endpoint (the same one their own search box
uses). Powers the "type MS, see MSFT" autocomplete in the frontend.
"""

import requests

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
        resp = requests.get(YAHOO_SEARCH_URL, params=params, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    # Only actual equities and ETFs work with this app's daily-price
    # prediction flow. Yahoo's search also returns options, futures,
    # indices, currencies, etc. — those would just produce a confusing
    # "no data found" error later, so filter them out here instead.
    ALLOWED_TYPES = {"EQUITY", "ETF"}

    results = []
    for quote in data.get("quotes", []):
        symbol = quote.get("symbol")
        name = quote.get("shortname") or quote.get("longname")
        exchange = quote.get("exchange")
        quote_type = quote.get("quoteType")
        if symbol and name and quote_type in ALLOWED_TYPES:
            results.append({"symbol": symbol, "name": name, "exchange": exchange})

    return results
