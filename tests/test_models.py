import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.features import calculate_rsi, engineer_features, prepare_train_test_data
from models.train import train_model


@pytest.fixture
def mock_market_df():
    """Generates synthetic daily market data for testing feature engineering."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(100)]
    np.random.seed(42)
    base_price = 150.0
    prices = base_price + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, size=100),
        "sentiment_score": np.random.uniform(-0.5, 0.5, size=100),
    })
    return df


def test_calculate_rsi():
    series = pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20, 22, 24, 23, 25, 27])
    rsi = calculate_rsi(series, period=14)
    assert not rsi.empty
    # Valid RSI values are bounded between 0 and 100
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_engineer_features(mock_market_df):
    df_feat = engineer_features(mock_market_df)
    assert not df_feat.empty
    assert "daily_return" in df_feat.columns
    assert "sma_7" in df_feat.columns
    assert "sma_21" in df_feat.columns
    assert "rsi_14" in df_feat.columns
    assert "volatility_5d" in df_feat.columns
    assert "target_up" in df_feat.columns
    # Check binary target
    assert set(df_feat["target_up"].unique()).issubset({0, 1})


def test_prepare_train_test_data(mock_market_df):
    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test_data(mock_market_df, train_ratio=0.8)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert "rsi_14" in feature_cols
