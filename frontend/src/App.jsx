import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return debounced;
}

function TickerSearch({ value, onChange, onSelect }) {
  const [query, setQuery] = useState(value);
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const boxRef = useRef(null);
  const debouncedQuery = useDebouncedValue(query, 250);

  useEffect(() => {
    function handleClickOutside(e) {
      if (boxRef.current && !boxRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 1) {
      setResults([]);
      return;
    }
    let cancelled = false;
    fetch(`${API_BASE}/search/${encodeURIComponent(debouncedQuery)}`)
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) {
          setResults(data.results || []);
          setOpen(true);
          setActiveIndex(-1);
        }
      })
      .catch(() => {
        if (!cancelled) setResults([]);
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedQuery]);

  function selectResult(r) {
    setQuery(r.symbol);
    onChange(r.symbol);
    onSelect(r);
    setOpen(false);
  }

  function handleKeyDown(e) {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      selectResult(results[activeIndex]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="ticker-search" ref={boxRef}>
      <input
        value={query}
        onChange={(e) => {
          const v = e.target.value.toUpperCase();
          setQuery(v);
          onChange(v);
        }}
        onFocus={() => results.length > 0 && setOpen(true)}
        onKeyDown={handleKeyDown}
        placeholder="Search company or ticker (e.g. MS, Apple)"
        autoComplete="off"
      />
      {open && results.length > 0 && (
        <ul className="ticker-dropdown">
          {results.map((r, i) => (
            <li
              key={r.symbol + i}
              className={i === activeIndex ? "active" : ""}
              onMouseDown={() => selectResult(r)}
              onMouseEnter={() => setActiveIndex(i)}
            >
              <span className="ticker-symbol">{r.symbol}</span>
              <span className="ticker-name">{r.name}</span>
              {r.exchange && <span className="ticker-exchange">{r.exchange}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [age, setAge] = useState(28);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedName, setSelectedName] = useState(null);

  const handlePredict = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const [predictRes, historyRes] = await Promise.all([
        fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, age: Number(age) }),
        }),
        fetch(`${API_BASE}/history/${ticker}`),
      ]);

      if (!predictRes.ok) {
        const errData = await predictRes.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      const predictData = await predictRes.json();
      const historyData = await historyRes.json();

      setResult(predictData);
      setHistory(
        historyData.dates.map((date, i) => ({
          date: date.slice(5),
          close: historyData.closes[i],
        }))
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, age]);

  const isNeutralUnclear =
    result && result.sentiment.label === "neutral" && result.sentiment.headlines.length === 0;

  return (
    <div className="page">
      <div className="container">
        <header className="hero">
          <h1>Stock Market Predictor</h1>
          <p className="subtitle">LSTM price prediction + live news sentiment + age-based advice</p>
        </header>

        <div className="search-card">
          <div className="form-row">
            <TickerSearch
              value={ticker}
              onChange={setTicker}
              onSelect={(r) => setSelectedName(r.name)}
            />
            <input
              className="age-input"
              type="number"
              value={age}
              min={1}
              max={120}
              onChange={(e) => setAge(e.target.value)}
              placeholder="Age"
            />
            <button onClick={handlePredict} disabled={loading || !ticker}>
              {loading ? (
                <>
                  <span className="spinner" /> Analyzing
                </>
              ) : (
                "Predict"
              )}
            </button>
          </div>
          {selectedName && <div className="selected-name">{selectedName}</div>}
        </div>

        {error && <div className="error-card">⚠ {error}</div>}

        {loading && !result && (
          <div className="card skeleton-card">
            <div className="skeleton-line short" />
            <div className="skeleton-line" />
            <div className="skeleton-line" />
          </div>
        )}

        {result && (
          <>
            <div className="card stat-card">
              <div className="stat-row">
                <div className="stat">
                  <div className="stat-label">Current Price</div>
                  <div className="stat-value">${result.current_price}</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Predicted Next</div>
                  <div
                    className={`stat-value ${
                      result.predicted_next_price >= result.current_price ? "up" : "down"
                    }`}
                  >
                    ${result.predicted_next_price}
                    <span className="delta">
                      {result.predicted_next_price >= result.current_price ? "▲" : "▼"}{" "}
                      {Math.abs(result.recommendation.predicted_change_pct)}%
                    </span>
                  </div>
                </div>
                <div className="stat">
                  <div className="stat-label">Recommendation</div>
                  <div className={`badge ${result.recommendation.recommendation}`}>
                    {result.recommendation.recommendation}
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="stat-label" style={{ marginBottom: 8 }}>
                Past 30 Days
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={history}>
                  <defs>
                    <linearGradient id="lineColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6d8bff" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#6d8bff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262a3a" />
                  <XAxis dataKey="date" stroke="#8b8fa3" fontSize={12} />
                  <YAxis stroke="#8b8fa3" fontSize={12} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1d29",
                      border: "1px solid #2a2d3a",
                      borderRadius: 8,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#6d8bff"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="stat-label">Why</div>
              <p>{result.recommendation.reason}</p>
              <div className="stat-label" style={{ marginTop: 14 }}>
                News Sentiment
              </div>
              {isNeutralUnclear ? (
                <p className="muted">No recent news found for this ticker — defaulted to neutral.</p>
              ) : (
                <p>
                  <span className={`sentiment-tag ${result.sentiment.label}`}>
                    {result.sentiment.label}
                  </span>{" "}
                  (score: {result.sentiment.average_compound})
                </p>
              )}
            </div>

            <div className="card">
              <div className="stat-label">Advice for Your Age Group</div>
              <p>{result.age_based_advice}</p>
            </div>

            <p className="disclaimer">
              This is a portfolio/demo project, not financial advice. Predictions are based on a
              single-day LSTM forecast and simple sentiment scoring.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
