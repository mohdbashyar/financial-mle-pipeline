import pytest
from unittest.mock import patch, MagicMock
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_store_news_in_chroma():
    """Test that store_news_in_chroma correctly processes input."""
    with patch("models.rag.collection") as mock_collection:
        mock_collection.upsert = MagicMock()
        
        from models.rag import store_news_in_chroma
        
        news_data = [
            {"title": "Stock surges 10%", "publisher": "Yahoo Finance", "uuid": "123"},
            {"content": {"title": "Company misses earnings"}, "uuid": "456"}
        ]
        
        store_news_in_chroma("AAPL", news_data)
        
        # Check that upsert was called with the correct extracted data
        mock_collection.upsert.assert_called_once()
        args, kwargs = mock_collection.upsert.call_args
        
        assert len(kwargs["documents"]) == 2
        assert "Stock surges 10%" in kwargs["documents"]
        assert "Company misses earnings" in kwargs["documents"]
        
        assert kwargs["metadatas"][0]["ticker"] == "AAPL"
        assert kwargs["ids"] == ["123", "456"]

def test_market_query_endpoint():
    """Test the /v1/market-query FastAPI endpoint."""
    with patch("api.main.query_rag") as mock_query_rag:
        mock_query_rag.return_value = {
            "answer": "This is a mock answer about Apple.",
            "sources": ["http://source.com/news"]
        }
        
        response = client.post("/v1/market-query", json={"query": "Why did AAPL go up?", "ticker": "AAPL"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a mock answer about Apple."
        assert len(data["sources"]) == 1
        assert data["sources"][0] == "http://source.com/news"
        
        mock_query_rag.assert_called_once_with(query="Why did AAPL go up?", ticker="AAPL")
