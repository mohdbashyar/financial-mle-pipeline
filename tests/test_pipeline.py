import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data_pipeline.db import Base, MarketData
from data_pipeline.fetch_data import fetch_stock_data, save_market_data_to_db
from data_pipeline.sentiment import analyze_headline_sentiment


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_sentiment_heuristic():
    pos_score = analyze_headline_sentiment("Company reports record profits and high growth")
    assert pos_score > 0

    neg_score = analyze_headline_sentiment("Stock plunges after massive loss and downgrade warning")
    assert neg_score < 0


def test_fetch_stock_data():
    df = fetch_stock_data("AAPL", period="5d")
    assert not df.empty
    assert "Date" in df.columns
    assert "Close" in df.columns
    assert "Sentiment_Score" in df.columns
    assert df["Ticker"].iloc[0] == "AAPL"


def test_save_market_data_to_db(in_memory_db):
    df = fetch_stock_data("AAPL", period="5d")
    count = save_market_data_to_db(df, session=in_memory_db)
    assert count > 0

    # Query DB to verify
    records = in_memory_db.query(MarketData).filter(MarketData.ticker == "AAPL").all()
    assert len(records) == len(df)

    # Test deduplication on re-save
    count_again = save_market_data_to_db(df, session=in_memory_db)
    assert count_again == len(df)
    records_after = in_memory_db.query(MarketData).filter(MarketData.ticker == "AAPL").all()
    assert len(records_after) == len(df)
