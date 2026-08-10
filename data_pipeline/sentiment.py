import logging
from typing import List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Lazy loader for HuggingFace FinBERT sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading HuggingFace FinBERT model (ProsusAI/finbert)...")
            _sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            logger.warning(f"Could not load FinBERT transformer pipeline: {e}. Falling back to basic sentiment heuristic.")
            _sentiment_pipeline = False
    return _sentiment_pipeline


def analyze_headline_sentiment(headline: str) -> float:
    """Classifies a financial headline and returns a sentiment score between -1.0 and +1.0."""
    if not headline or not headline.strip():
        return 0.0

    pipe = get_sentiment_pipeline()
    if pipe:
        try:
            result = pipe(headline)[0]
            label = result.get("label", "").lower()
            score = float(result.get("score", 0.0))

            if label == "positive":
                return score
            elif label == "negative":
                return -score
            return 0.0
        except Exception as e:
            logger.error(f"Error evaluating sentiment with FinBERT: {e}")

    # Basic fallback heuristic if FinBERT pipeline is not available
    headline_lower = headline.lower()
    positive_words = {"bullish", "surge", "growth", "profit", "record", "upgrade", "gain", "high", "rise", "dividend"}
    negative_words = {"bearish", "plunge", "loss", "drop", "downgrade", "fall", "risk", "lawsuit", "decline", "warn"}

    pos_count = sum(1 for word in positive_words if word in headline_lower)
    neg_count = sum(1 for word in negative_words if word in headline_lower)

    if pos_count > neg_count:
        return 0.5
    elif neg_count > pos_count:
        return -0.5
    return 0.0


def fetch_ticker_news_sentiment(ticker_symbol: str) -> float:
    """Fetches recent news headlines for a ticker, stores them in ChromaDB, and computes an average sentiment score."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news:
            return 0.0

        # Store raw news items in ChromaDB for RAG queries
        try:
            from models.rag import store_news_in_chroma
            store_news_in_chroma(ticker_symbol, news)
        except Exception as e:
            logger.error(f"Error storing news in ChromaDB: {e}")

        scores: List[float] = []
        for item in news:
            # yfinance news structure: 'title' or nested 'content'
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                score = analyze_headline_sentiment(title)
                scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            return round(avg_score, 4)
    except Exception as e:
        logger.warning(f"Failed to fetch news sentiment for ticker {ticker_symbol}: {e}")

    return 0.0
