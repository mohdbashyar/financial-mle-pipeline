import sys
import os

# Ensure repository root directory is in sys.path for cloud deployment module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Configuration & Modern Theme Styling
st.set_page_config(
    page_title="Financial MLE Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich modern aesthetics & high-contrast metric cards
st.markdown("""
    <style>
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            padding: 18px !important;
            border-radius: 14px !important;
            border: 1px solid #334155 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: #38BDF8 !important;
            transform: translateY(-2px);
            transition: all 0.2s ease-in-out;
        }
        [data-testid="stMetricLabel"] {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #CBD5E1 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricValue"] div {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }
        .trend-up {
            color: #4ADE80 !important;
            font-weight: bold;
        }
        .trend-down {
            color: #F87171 !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def fetch_prediction(ticker: str):
    try:
        res = requests.post(f"{API_URL}/predict", json={"ticker": ticker}, timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Direct fallback for Standalone Cloud Deployment (e.g. Streamlit Community Cloud)
    try:
        from data_pipeline.fetch_data import fetch_stock_data, save_market_data_to_db
        from models.features import load_data_from_db, engineer_features
        from models.train import train_model, ARTIFACT_DIR
        import pickle

        df = load_data_from_db(ticker=ticker)
        if df.empty or len(df) < 30:
            raw_df = fetch_stock_data(ticker, period="1y")
            save_market_data_to_db(raw_df)
            df = load_data_from_db(ticker=ticker)

        df_feat = engineer_features(df)
        if not df_feat.empty:
            model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
            meta_path = os.path.join(ARTIFACT_DIR, "metadata.pkl")

            if os.path.exists(model_path) and os.path.exists(meta_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
            else:
                model, metadata = train_model(ticker=ticker)

            feature_cols = metadata.get("feature_cols", [
                "daily_return", "sma_7", "sma_21", "rsi_14",
                "volatility_5d", "return_lag1", "return_lag2", "sentiment_score"
            ])
            latest_row = df_feat.iloc[-1]
            X_latest = latest_row[feature_cols].values.reshape(1, -1)

            pred_class = int(model.predict(X_latest)[0])
            prob = model.predict_proba(X_latest)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

            return {
                "ticker": ticker,
                "prediction": "UP" if pred_class == 1 else "DOWN",
                "confidence": round(float(prob[pred_class]), 4),
                "model_version": metadata.get("model_version", "v1.0") + " (Cloud)",
                "latest_price": float(latest_row["close"]),
                "sentiment_score": float(latest_row["sentiment_score"]),
                "date": str(latest_row["date"].strftime("%Y-%m-%d")),
            }
    except Exception as e:
        st.error(f"Error executing prediction engine: {e}")
    return None


def fetch_historical_data(ticker: str):
    try:
        res = requests.get(f"{API_URL}/data/{ticker}", timeout=3)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                return df
    except Exception:
        pass

    # Direct fallback for Standalone Cloud Deployment
    try:
        from data_pipeline.fetch_data import fetch_stock_data, save_market_data_to_db
        from models.features import load_data_from_db

        df = load_data_from_db(ticker=ticker)
        if df.empty:
            raw_df = fetch_stock_data(ticker, period="6mo")
            save_market_data_to_db(raw_df)
            df = load_data_from_db(ticker=ticker)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as e:
        st.error(f"Error loading historical market data: {e}")
    return pd.DataFrame()


# Header Banner
st.title("⚡ Financial Machine Learning Engineering Pipeline")
st.markdown("Automated OHLCV market ingestion, FinBERT news sentiment inference, and predictive trend modeling.")

# Sidebar Controls
st.sidebar.header("🎯 Market Controls")
default_tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
selected_ticker = st.sidebar.selectbox("Select Benchmark Asset", default_tickers)
custom_ticker = st.sidebar.text_input("Or enter custom ticker", "").strip().upper()

active_ticker = custom_ticker if custom_ticker else selected_ticker

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")

health = check_api_health()
if health:
    st.sidebar.success(f"FastAPI Backend: Online ({health.get('status')})")
    st.sidebar.info(f"Database: {'Connected' if health.get('database_connected') else 'Offline'}")
    st.sidebar.info(f"ML Model: {'Loaded' if health.get('model_loaded') else 'On-Demand'}")
else:
    st.sidebar.info("Execution Mode: Standalone Direct Cloud Pipeline")

if st.sidebar.button("🔄 Refresh Data & Predict", use_container_width=True):
    st.cache_data.clear()

# Main App Body
with st.spinner(f"Fetching predictions and market telemetry for {active_ticker}..."):
    prediction_data = fetch_prediction(active_ticker)
    df_history = fetch_historical_data(active_ticker)

if prediction_data:
    pred = prediction_data.get("prediction", "N/A")
    conf = prediction_data.get("confidence", 0.0) * 100
    price = prediction_data.get("latest_price", 0.0)
    sentiment = prediction_data.get("sentiment_score", 0.0)

    # Top KPI Dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"Latest Price ({active_ticker})",
            value=f"${price:,.2f}",
        )

    with col2:
        trend_class = "trend-up" if pred == "UP" else "trend-down"
        st.metric(
            label="Predicted Price Move (T+1)",
            value=f"{'🟢' if pred == 'UP' else '🔴'} {pred}",
            delta=f"{conf:.1f}% Confidence",
        )

    with col3:
        st.metric(
            label="FinBERT News Sentiment",
            value=f"{sentiment:+.2f}",
            delta="Positive" if sentiment > 0 else ("Negative" if sentiment < 0 else "Neutral"),
        )

    with col4:
        st.metric(
            label="Model Artifact Version",
            value=prediction_data.get("model_version", "v1.0"),
            delta="RandomForest / XGBoost",
        )

# Historical Charts Section
st.markdown("---")
st.subheader(f"📊 Technical & Sentiment Analysis: {active_ticker}")

if not df_history.empty:
    # Compute moving averages for chart
    df_history["sma_7"] = df_history["close"].rolling(7).mean()
    df_history["sma_21"] = df_history["close"].rolling(21).mean()

    chart_type = st.radio("Chart Type", ["Line Chart with Moving Averages", "Candlestick"], horizontal=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{active_ticker} Price Action & Moving Averages", "Daily News Sentiment Score")
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df_history["date"],
                open=df_history["open"],
                high=df_history["high"],
                low=df_history["low"],
                close=df_history["close"],
                name="OHLC",
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=df_history["date"], y=df_history["close"], mode="lines", name="Close Price", line=dict(color="#38BDF8", width=2)),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=df_history["date"], y=df_history["sma_7"], mode="lines", name="7-Day SMA", line=dict(color="#FBBF24", width=1.5, dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_history["date"], y=df_history["sma_21"], mode="lines", name="21-Day SMA", line=dict(color="#A855F7", width=1.5, dash="dot")),
        row=1, col=1
    )

    # Sentiment subplot
    colors = ["#4ADE80" if s > 0 else ("#F87171" if s < 0 else "#94A3B8") for s in df_history["sentiment_score"]]
    fig.add_trace(
        go.Bar(x=df_history["date"], y=df_history["sentiment_score"], name="Sentiment", marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table Toggle
    with st.expander("🔍 View Raw Historical Telemetry Table"):
        st.dataframe(df_history.sort_values("date", ascending=False), use_container_width=True)
else:
    st.info("No historical data displayed yet. Please launch the FastAPI server or trigger data ingestion.")
