from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PredictRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock or crypto ticker symbol")


class PredictResponse(BaseModel):
    ticker: str
    prediction: str  # "UP" or "DOWN"
    confidence: float
    model_version: str
    latest_price: float
    sentiment_score: float
    date: str


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool


class MarketDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    sentiment_score: float


class HistoryResponse(BaseModel):
    ticker: str
    total_records: int
    data: List[MarketDataPoint]
