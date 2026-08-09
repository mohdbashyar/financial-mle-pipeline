import logging
import argparse
from data_pipeline.fetch_data import fetch_stock_data, save_market_data_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]


def run_ingestion_pipeline(tickers=None, period="2y"):
    """Runs the full data ingestion pipeline for the specified tickers."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    logger.info(f"Starting data ingestion pipeline for tickers: {tickers}")
    total_processed = 0

    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period=period)
            count = save_market_data_to_db(df)
            logger.info(f"Successfully processed {ticker}: {count} records in DB.")
            total_processed += count
        except Exception as e:
            logger.error(f"Failed to ingest data for {ticker}: {e}")

    logger.info(f"Pipeline execution finished. Total DB operations: {total_processed}")
    return total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial MLE Data Ingestion Pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to ingest")
    parser.add_argument("--period", type=str, default="2y", help="Historical timeframe (e.g. 1mo, 1y, 2y)")

    args = parser.parse_args()
    run_ingestion_pipeline(tickers=args.tickers, period=args.period)
