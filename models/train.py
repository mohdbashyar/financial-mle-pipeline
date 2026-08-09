import os
import pickle
import logging
import argparse
import mlflow
import mlflow.sklearn
try:
    import mlflow.xgboost
except ImportError:
    pass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.features import load_data_from_db, prepare_train_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def train_model(ticker: str = "AAPL", n_estimators: int = 100, max_depth: int = 5, model_type: str = "rf"):
    """Trains a machine learning model on historical ticker data and tracks with MLflow."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    logger.info(f"Loading data from DB for ticker: {ticker}...")
    df = load_data_from_db(ticker=ticker)
    if df.empty or len(df) < 30:
        raise ValueError(f"Insufficient data in DB for ticker '{ticker}'. Please run data ingestion first.")

    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test_data(df)
    logger.info(f"Data prepared: Train set size = {len(X_train)}, Test set size = {len(X_test)}")

    # Set MLflow experiment name
    mlflow.set_experiment(f"Financial_MLE_{ticker}")

    with mlflow.start_run():
        if model_type == "xgb":
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            except ImportError:
                logger.warning("XGBoost not installed. Falling back to RandomForestClassifier.")
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model_type = "rf"
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        logger.info(f"Model ({model_type}) Evaluation Results:")
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"ROC-AUC:   {auc:.4f}")

        # MLflow Logging
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("train_samples", len(X_train))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        if model_type == "xgb":
            try:
                mlflow.xgboost.log_model(model, "model")
            except Exception:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
                )
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save model locally for FastAPI serving
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        meta_path = os.path.join(ARTIFACT_DIR, "metadata.pkl")

        metadata = {
            "ticker": ticker,
            "feature_cols": feature_cols,
            "accuracy": acc,
            "f1_score": f1,
            "model_version": "v1.0",
        }

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved trained model artifact to {model_path}")
        return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Financial MLE Model")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to train on")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Max tree depth")
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb"], help="Model architecture")

    args = parser.parse_args()
    train_model(ticker=args.ticker, n_estimators=args.n_estimators, max_depth=args.max_depth, model_type=args.model_type)
