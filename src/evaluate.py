"""
Model evaluation module.
Loads trained inference pipeline and evaluates its performance on validation/test data.
"""
import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data import load_config, load_data, split_data


def load_model(model_path: str = "models/model.joblib"):
    """Loads saved model pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def evaluate_model(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Evaluates the saved model pipeline against held-out test data.
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    reports_cfg = config["reports"]

    model_path = model_cfg.get("model_save_path", "models/model.joblib")
    print(f"Loading pipeline from {model_path}...")
    pipeline = load_model(model_path)

    # Load processed data
    data_path = data_cfg.get("processed_path", data_cfg["raw_path"])
    df = load_data(data_path)

    # Split to get held-out test set
    _, X_test, _, y_test = split_data(
        df=df,
        target_column=data_cfg["target_column"],
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "test_samples": len(X_test)
    }

    print("--- Evaluation Results ---")
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      ${rmse:,.2f}")
    print(f"MAE:       ${mae:,.2f}")
    print(f"Samples:   {len(X_test)}")

    metrics_path = reports_cfg.get("metrics_path", "reports/metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate house price prediction pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    evaluate_model(args.config)


if __name__ == "__main__":
    main()

