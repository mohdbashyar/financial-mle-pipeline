import pytest
import pandas as pd
from app.main import check_api_health, fetch_prediction, fetch_historical_data


def test_streamlit_app_functions_importable():
    # Verify functions are callable and handle offline API gracefully
    health = check_api_health()
    # Offline API should return None without crashing
    assert health is None or isinstance(health, dict)

    pred = fetch_prediction("NONEXISTENT_TICKER_123")
    assert pred is None or isinstance(pred, dict)

    df_hist = fetch_historical_data("NONEXISTENT_TICKER_123")
    assert isinstance(df_hist, pd.DataFrame)
