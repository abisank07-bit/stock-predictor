"""
search.py
Worldwide ticker lookup for the autocomplete box.

Uses Yahoo Finance's public search endpoint (the same one their own search
box uses), which covers listed companies and ETFs on exchanges across the
world. Type a company name ("toyota", "infosys") or the start of a ticker
("ms") and get matching listings, each labelled with its exchange and country.
"""

import threading
import time

import requests

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

HEADERS = {
    # Yahoo's endpoint blocks requests with no user-agent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Ask Yahoo for plenty of matches (some get filtered out below) and show the best few.
FETCH_COUNT = 20
RESULT_LIMIT = 12
MAX_QUERY_LEN = 40

# Only these work with the app's daily-price prediction flow. Yahoo's search
# also returns options, futures, indices, currencies, crypto and mutual funds.
ALLOWED_TYPES = {"EQUITY", "ETF"}

# Yahoo marks non-US listings with an exchange suffix: INFY.NS, 7203.T, SAP.DE ...
# US listings have no suffix. Used only to show the country next to each result.
SUFFIX_COUNTRY = {
    # Americas
    "TO": "Canada", "V": "Canada", "CN": "Canada", "NE": "Canada",
    "MX": "Mexico", "SA": "Brazil", "BA": "Argentina", "SN": "Chile",
    # Europe
    "L": "UK", "IL": "UK",
    "DE": "Germany", "F": "Germany", "BE": "Germany", "DU": "Germany",
    "HM": "Germany", "HA": "Germany", "MU": "Germany", "SG": "Germany",
    "PA": "France", "AS": "Netherlands", "BR": "Belgium", "LS": "Portugal",
    "MI": "Italy", "MC": "Spain", "VI": "Austria", "SW": "Switzerland",
    "ST": "Sweden", "HE": "Finland", "CO": "Denmark", "OL": "Norway",
    "IC": "Iceland", "WA": "Poland", "PR": "Czechia", "AT": "Greece",
    "IS": "Turkey", "TA": "Israel",
    # Asia-Pacific
    "NS": "India", "BO": "India", "T": "Japan", "HK": "Hong Kong",
    "SS": "China", "SZ": "China", "TW": "Taiwan", "TWO": "Taiwan",
    "KS": "South Korea", "KQ": "South Korea", "SI": "Singapore",
    "KL": "Malaysia", "BK": "Thailand", "JK": "Indonesia",
    "AX": "Australia", "NZ": "New Zealand",
    # Middle East & Africa
    "JO": "South Africa", "CA": "Egypt", "SR": "Saudi Arabia", "QA": "Qatar",
}

# Small in-memory cache so repeated lookups (typing, backspacing, many
# visitors) don't hammer Yahoo, which rate-limits cloud servers.
CACHE_TTL_SECONDS = 600
CACHE_MAX_ENTRIES = 300
_cache = {}
_cache_lock = threading.Lock()


def _cache_get(key):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        if now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
        del _cache[key]
    return None


def _cache_put(key, value):
    with _cache_lock:
        if key not in _cache and len(_cache) >= CACHE_MAX_ENTRIES:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest]
        _cache[key] = (time.monotonic(), value)


def _fetch_quotes(query):
    """One call to Yahoo's search endpoint. Raises on any network/HTTP/parse problem."""
    params = {
        "q": query,
        "quotesCount": FETCH_COUNT,
        "newsCount": 0,
    }
    resp = requests.get(YAHOO_SEARCH_URL, params=params, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    quotes = resp.json().get("quotes")
    if not isinstance(quotes, list):
        # A 200 without a proper results list means Yahoo is throttling or
        # misbehaving, not that nothing matched, so treat it as a failure.
        raise ValueError("unexpected response from Yahoo")
    return quotes


def _exchange_label(quote, symbol):
    """'NASDAQ' for US listings, 'NSE · India' for INFY.NS, and so on."""
    exchange = quote.get("exchDisp") or quote.get("exchange") or ""
    suffix = symbol.rsplit(".", 1)[1].upper() if "." in symbol else ""
    country = SUFFIX_COUNTRY.get(suffix)
    if exchange and country:
        return f"{exchange} · {country}"
    return exchange or country or None


def _rank(items, query):
    """Put exact ticker matches first, then tickers/names starting with the query."""
    q = query.strip().upper()

    def rank(item):
        symbol = item["symbol"].upper()
        base = symbol.split(".")[0]  # INFY.NS -> INFY
        if q in (symbol, base):
            return 0
        if symbol.startswith(q) or item["name"].upper().startswith(q):
            return 1
        return 2

    # sorted() is stable, so Yahoo's own relevance order breaks ties
    return sorted(items, key=rank)


def _clean(quotes, query):
    seen = set()
    items = []
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        symbol = quote.get("symbol")
        name = quote.get("longname") or quote.get("shortname")
        if not symbol or not name:
            continue
        if quote.get("quoteType") not in ALLOWED_TYPES:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        items.append(
            {"symbol": symbol, "name": name, "exchange": _exchange_label(quote, symbol)}
        )
    return _rank(items, query)


def search_tickers(query: str, limit: int = RESULT_LIMIT):
    """
    Returns a list of {symbol, name, exchange} matches for a company name or
    part of a ticker, from stock exchanges worldwide, e.g.
    "ms" -> MS, MSFT, ...   "infosys" -> INFY.NS, INFY, ...
    """
    q = (query or "").strip()[:MAX_QUERY_LEN]
    if not q:
        return []

    key = q.lower()
    cached = _cache_get(key)
    if cached is not None:
        return [dict(r) for r in cached[:limit]]

    try:
        quotes = _fetch_quotes(q)
    except Exception:
        # Network error, rate limit, bad JSON ... autocomplete just shows
        # nothing this time. Failures are never cached, so the next keystroke retries.
        return []

    results = _clean(quotes, q)
    if results:
        # Only remember real answers. An empty list may just be Yahoo having a
        # bad moment, and we don't want to hide a query for 10 minutes because of it.
        _cache_put(key, results)
    return [dict(r) for r in results[:limit]]
