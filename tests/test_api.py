import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database_connected" in data
    assert "model_loaded" in data


def test_predict_endpoint():
    response = client.post("/predict", json={"ticker": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["prediction"] in ["UP", "DOWN"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "latest_price" in data
    assert "sentiment_score" in data


def test_get_historical_data_endpoint():
    response = client.get("/data/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["total_records"] > 0
    assert isinstance(data["data"], list)
    first_point = data["data"][0]
    assert "close" in first_point
    assert "sentiment_score" in first_point
