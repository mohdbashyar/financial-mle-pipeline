import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from data_pipeline.db import DATABASE_URL, MarketData, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_from_db(ticker: str = None) -> pd.DataFrame:
    """Loads market data records from the database into a DataFrame."""
    init_db()
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM market_data"
    if ticker:
        query += f" WHERE ticker = '{ticker.upper()}'"
    query += " ORDER BY date ASC"

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"Error executing SQL query: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No database records found for query: {query}")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers technical indicators, sentiment features, and next-day price trend target."""
    if df.empty or len(df) < 30:
        logger.warning("Insufficient data points for feature engineering.")
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True).copy()

    # Price returns
    df["daily_return"] = df["close"].pct_change()

    # Moving Averages
    df["sma_7"] = df["close"].rolling(window=7).mean()
    df["sma_21"] = df["close"].rolling(window=21).mean()

    # RSI
    df["rsi_14"] = calculate_rsi(df["close"], period=14)

    # Rolling Volatility (5-day standard deviation of daily return)
    df["volatility_5d"] = df["daily_return"].rolling(window=5).std()

    # Lagged Returns
    df["return_lag1"] = df["daily_return"].shift(1)
    df["return_lag2"] = df["daily_return"].shift(2)

    # Sentiment Score (already present or fallback to 0.0)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    # Target: Next Day Price Trend (1 if next day close > today close else 0)
    df["target_next_close"] = df["close"].shift(-1)
    df["target_up"] = (df["target_next_close"] > df["close"]).astype(int)

    # Drop rows with NaN (due to rolling windows and last row shift)
    df_clean = df.dropna().reset_index(drop=True).copy()
    return df_clean


def prepare_train_test_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Splits engineered DataFrame into chronological train/test sets."""
    df_features = engineer_features(df)
    if df_features.empty:
        raise ValueError("Engineered DataFrame is empty.")

    feature_cols = [
        "daily_return",
        "sma_7",
        "sma_21",
        "rsi_14",
        "volatility_5d",
        "return_lag1",
        "return_lag2",
        "sentiment_score",
    ]

    X = df_features[feature_cols]
    y = df_features["target_up"]

    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test, feature_cols
