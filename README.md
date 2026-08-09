# ⚡ Financial Machine Learning Engineering (MLE) Pipeline

An end-to-end production-grade financial ML pipeline featuring automated market data ingestion (`yfinance`), FinBERT news sentiment inference (`transformers`), experiment tracking (`MLflow`), high-performance model serving (`FastAPI`), interactive visual dashboard (`Streamlit`), and full container orchestration (`Docker Compose`).

---

## 🏛️ System Architecture

```
                                               ┌───────────────────────┐
┌──────────────────┐                            │ Hugging Face FinBERT  │
│  yfinance API    │ ──> Price Data (OHLCV) ──> │   (Transformer Model) │
└──────────────────┘                            └──────────┬────────────┘
                                                           │
┌──────────────────┐                                       ▼
│  Yahoo News API  │ ──> Headlines / News ────> [ Calculated ] ──> [ SQLite / Postgres ]
└──────────────────┘                            [ Sentiment  ]     [   Database    ]
                                                [   Score    ]
                                                           │
                                                           ▼
                                                ┌──────────────────────┐       ┌──────────────────┐
                                                │ Feature Engineering  │ ────> │ MLflow Tracking  │
                                                │ (SMA, RSI, Target)   │       │ & Model Registry │
                                                └──────────┬───────────┘       └──────────────────┘
                                                           │
                                                           ▼
                                                ┌──────────────────────┐
                                                │ FastAPI Endpoint     │
                                                │ (/predict, /data)    │
                                                └──────────┬───────────┘
                                                           │
                                                           ▼
                                                ┌──────────────────────┐
                                                │ Streamlit Dashboard  │
                                                └──────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Local Setup (Without Docker)

```bash
# Activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# 1. Run Data Ingestion Pipeline (Fetches OHLCV & News Sentiment)
python -m data_pipeline.run_pipeline --tickers AAPL MSFT NVDA SPY BTC-USD --period 2y

# 2. Train Model & Track with MLflow
python -m models.train --ticker AAPL --model_type rf

# 3. Launch FastAPI REST Server
uvicorn api.main:app --reload --port 8000

# 4. Launch Streamlit Web Dashboard (In another terminal)
streamlit run app/main.py
```

Open `http://localhost:8501` to access the interactive dashboard, and `http://localhost:8000/docs` to view interactive OpenAPI docs.

---

## 🐳 Docker Deployment (Single Command)

```bash
docker-compose up --build
```

Spins up:
- **PostgreSQL Database**: Port `5432`
- **FastAPI Model Server**: `http://localhost:8000`
- **Streamlit Web UI**: `http://localhost:8501`

---

## 🧪 Running Tests

```bash
pytest tests/
```
