import os
import pickle
import logging
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.schemas import PredictRequest, PredictResponse, HealthResponse, HistoryResponse, MarketDataPoint
from data_pipeline.db import get_db_session, MarketData, engine
from data_pipeline.fetch_data import fetch_stock_data, save_market_data_to_db
from models.features import engineer_features, load_data_from_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial MLE Pipeline Serving API",
    description="Production REST API for Stock Price Trend Predictions & Market Data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.pkl")


def load_trained_model():
    """Loads saved model and metadata from artifact path."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(META_PATH, "rb") as f:
                metadata = pickle.load(f)
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading trained model artifact: {e}")
    return None, None


@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db_session)):
    """Health check endpoint to monitor database and model readiness."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Database connection health check failed: {e}")

    model, _ = load_trained_model()
    model_loaded = model is not None

    return HealthResponse(
        status="ok" if (db_connected and model_loaded) else "degraded",
        database_connected=db_connected,
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_trend(request: PredictRequest, db: Session = Depends(get_db_session)):
    """Accepts ticker symbol, calculates technical indicators, and returns binary price movement prediction."""
    ticker = request.ticker.strip().upper()
    logger.info(f"Received prediction request for ticker: {ticker}")

    # Fetch records from database
    df = load_data_from_db(ticker=ticker)
    if df.empty or len(df) < 30:
        logger.info(f"Insufficient DB data for {ticker}. Ingesting historical data...")
        try:
            raw_df = fetch_stock_data(ticker, period="1y")
            save_market_data_to_db(raw_df, session=db)
            df = load_data_from_db(ticker=ticker)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch market data for '{ticker}': {str(e)}")

    if df.empty or len(df) < 30:
        raise HTTPException(status_code=400, detail=f"Insufficient history to compute indicators for '{ticker}'.")

    # Feature engineering
    df_features = engineer_features(df)
    if df_features.empty:
        raise HTTPException(status_code=500, detail="Feature engineering produced an empty dataset.")

    latest_row = df_features.iloc[-1]

    # Load trained model
    model, metadata = load_trained_model()
    if model is None:
        # Train an on-the-fly model if no serialized model exists yet
        logger.warning("No pre-trained model found. Training default model on demand...")
        from models.train import train_model
        model, metadata = train_model(ticker=ticker)

    feature_cols = metadata.get("feature_cols", [
        "daily_return", "sma_7", "sma_21", "rsi_14",
        "volatility_5d", "return_lag1", "return_lag2", "sentiment_score"
    ])

    # Extract feature values for prediction
    X_latest = latest_row[feature_cols].values.reshape(1, -1)

    prediction_class = int(model.predict(X_latest)[0])
    probabilities = model.predict_proba(X_latest)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    confidence = float(probabilities[prediction_class])

    prediction_str = "UP" if prediction_class == 1 else "DOWN"

    return PredictResponse(
        ticker=ticker,
        prediction=prediction_str,
        confidence=round(confidence, 4),
        model_version=metadata.get("model_version", "v1.0"),
        latest_price=float(latest_row["close"]),
        sentiment_score=float(latest_row["sentiment_score"]),
        date=str(latest_row["date"].strftime("%Y-%m-%d")),
    )


@app.get("/data/{ticker}", response_model=HistoryResponse)
def get_historical_data(ticker: str, db: Session = Depends(get_db_session)):
    """Returns historical market data and sentiment for UI charting."""
    ticker_clean = ticker.strip().upper()
    records = (
        db.query(MarketData)
        .filter(MarketData.ticker == ticker_clean)
        .order_by(MarketData.date.asc())
        .all()
    )

    if not records:
        # Ingest automatically if not in DB
        try:
            raw_df = fetch_stock_data(ticker_clean, period="6mo")
            save_market_data_to_db(raw_df, session=db)
            records = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker_clean)
                .order_by(MarketData.date.asc())
                .all()
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No data available for ticker '{ticker_clean}': {e}")

    data_points = [
        MarketDataPoint(
            date=str(r.date),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
            sentiment_score=float(r.sentiment_score),
        )
        for r in records
    ]

    return HistoryResponse(
        ticker=ticker_clean,
        total_records=len(data_points),
        data=data_points,
    )
