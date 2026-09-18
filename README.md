# Stock Market Predictor (v2 — Full Stack)

A rebuild of your original school project, upgraded to a full-stack app with a real
LSTM price prediction model, live news sentiment analysis, and age-based advice.

## Architecture

```
stock-predictor/
├── backend/           FastAPI + PyTorch LSTM + sentiment
│   ├── app/
│   │   ├── main.py        API endpoints
│   │   ├── data.py        yfinance data fetching + preprocessing
│   │   ├── model.py       LSTM model, training, prediction
│   │   ├── sentiment.py   News headline sentiment (VADER)
│   │   └── advice.py      Age-based advice + buy/sell/hold logic
│   └── requirements.txt
└── frontend/           React (Vite) dashboard
    └── src/
        ├── App.jsx         Main dashboard UI
        └── index.css
```

## How it works

1. **Data**: `yfinance` pulls real historical OHLCV data for any ticker (e.g. `AAPL`, `INFY.NS`, `ICICIBANK.NS`)
2. **Prediction**: A PyTorch LSTM is trained on the last 2 years of closing prices (60-day lookback window) to predict the next day's close
3. **Sentiment**: Recent news headlines for the ticker are scored with VADER sentiment analysis
4. **Recommendation**: Price prediction + sentiment are fused into a buy/sell/hold call — with the reasoning shown, not just the answer
5. **Personalization**: Age input maps to general financial advice (younger = more growth-oriented, older = more capital preservation)

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API will be live at `http://localhost:8000`. Try `http://localhost:8000/docs` for interactive API docs (FastAPI gives you this for free — mention it in your portfolio write-up).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App will be live at `http://localhost:5173`.

## First run notes

- The first `/predict` call for a new ticker will auto-train a model (takes ~10-30 seconds depending on your machine). After that, the trained model is cached in `backend/app/saved_models/` and reused.
- If you want to pre-train before demoing, hit `POST /train/{ticker}` first (e.g. via the `/docs` page or `curl`).
- For Indian stocks, append the exchange suffix: `INFY.NS`, `ICICIBANK.NS`, `RELIANCE.NS`.

## Ideas to extend further (good for a portfolio write-up)

- Swap VADER for **FinBERT** (finance-tuned sentiment) for more accurate news scoring
- Add a **backtesting** endpoint: run the model against past data and show predicted vs actual
- Add **confidence intervals** on the prediction instead of a single number
- Deploy: frontend to Vercel, backend to Render/Railway; store trained models in cloud storage (S3) instead of local disk
- Add auth + a database (PostgreSQL) to save each user's prediction history

## Known limitations (worth stating honestly in interviews)

- This predicts the *next single day's* closing price — it does not do multi-day forecasting
- Real stock prices are influenced by countless unmodeled factors; treat this as a learning/demo project, not investment advice
- The sentiment-price fusion logic is a simple rule-based combination, not a jointly trained model — a good "future work" talking point
