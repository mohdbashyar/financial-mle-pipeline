import logging
from datetime import datetime, date
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from data_pipeline.db import MarketData, SessionLocal, init_db
from data_pipeline.sentiment import fetch_ticker_news_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_stock_data(ticker_symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetches historical OHLCV market data for a ticker using yfinance.

    :param ticker_symbol: Ticker symbol (e.g. 'AAPL', 'MSFT', 'BTC-USD')
    :param period: Timeframe to fetch (e.g. '1mo', '1y', '2y', 'max')
    :return: Cleaned Pandas DataFrame
    """
    logger.info(f"Fetching OHLCV data for ticker: {ticker_symbol} (period={period})...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker_symbol}'. Please check symbol.")

    # Ensure required columns exist
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in yfinance output.")

    df = df[cols].copy()

    # Convert DatetimeIndex to explicit Date column (date object)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Add metadata
    df["Ticker"] = ticker_symbol.upper()

    # Calculate current news sentiment score
    latest_sentiment = fetch_ticker_news_sentiment(ticker_symbol)
    logger.info(f"News sentiment score for {ticker_symbol}: {latest_sentiment}")
    df["Sentiment_Score"] = latest_sentiment

    return df


def save_market_data_to_db(df: pd.DataFrame, session: Session = None) -> int:
    """Upserts market data DataFrame rows into the database."""
    init_db()
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    added_count = 0
    updated_count = 0

    try:
        for _, row in df.iterrows():
            row_date = row["Date"]
            if isinstance(row_date, str):
                row_date = datetime.strptime(row_date, "%Y-%m-%d").date()

            ticker_str = str(row["Ticker"]).upper()

            # Check if record already exists
            existing = (
                session.query(MarketData)
                .filter(MarketData.date == row_date, MarketData.ticker == ticker_str)
                .first()
            )

            if existing:
                existing.open = float(row["Open"])
                existing.high = float(row["High"])
                existing.low = float(row["Low"])
                existing.close = float(row["Close"])
                existing.volume = float(row["Volume"])
                existing.sentiment_score = float(row["Sentiment_Score"])
                updated_count += 1
            else:
                record = MarketData(
                    date=row_date,
                    ticker=ticker_str,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    sentiment_score=float(row["Sentiment_Score"]),
                )
                session.add(record)
                added_count += 1

        session.commit()
        logger.info(f"Database sync complete: {added_count} rows added, {updated_count} rows updated.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving market data to database: {e}")
        raise e
    finally:
        if close_session:
            session.close()

    return added_count + updated_count


if __name__ == "__main__":
    test_ticker = "AAPL"
    data = fetch_stock_data(test_ticker, period="1mo")
    print(data.head())
    saved = save_market_data_to_db(data)
    print(f"Total rows processed: {saved}")
