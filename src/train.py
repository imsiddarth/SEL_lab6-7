"""
Model training and pipeline assembly module.
Trains a complete Scikit-learn Pipeline (Preprocessing + Estimator)
and saves it to disk for reproducible inference.
"""
import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data import load_config, load_data, preprocess_data, split_data, save_processed_data


def build_pipeline(model_config: Dict[str, Any]) -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline combining preprocessing and estimator.
    This guarantees zero data leakage and identical transformations during training and inference.
    """
    model_type = model_config.get("type", "random_forest")
    n_estimators = model_config.get("n_estimators", 100)
    max_depth = model_config.get("max_depth", 8)
    random_state = model_config.get("random_state", 42)
    
    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", estimator)
    ])
    return pipeline


def train_model(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Executes end-to-end model training:
    1. Loads configuration.
    2. Ingests raw data and cleans it.
    3. Splits data into train and test sets.
    4. Fits pipeline strictly on train data.
    5. Evaluates on test data.
    6. Persists trained pipeline and evaluation metrics.
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    reports_cfg = config["reports"]

    print("[1/5] Loading and cleaning raw dataset...")
    raw_df = load_data(data_cfg["raw_path"])
    clean_df = preprocess_data(raw_df, target_column=data_cfg["target_column"])
    save_processed_data(clean_df, data_cfg["processed_path"])

    print("[2/5] Splitting into train and test sets (stratified/random)...")
    X_train, X_test, y_train, y_test = split_data(
        df=clean_df,
        target_column=data_cfg["target_column"],
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("[3/5] Building and fitting Scikit-Learn Pipeline on training data...")
    pipeline = build_pipeline(model_cfg)
    pipeline.fit(X_train, y_train)

    print("[4/5] Evaluating pipeline performance on held-out test set...")
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test)
    }
    print(f"      Test Metrics -> RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | R2: {r2:.4f}")

    print("[5/5] Saving model pipeline and reports...")
    model_save_path = model_cfg.get("model_save_path", "models/model.joblib")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"      Saved complete pipeline to: {model_save_path}")

    # Save metrics JSON
    metrics_path = reports_cfg.get("metrics_path", "reports/metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"      Saved metrics to: {metrics_path}")

    # Save Markdown Evaluation Report
    eval_report_path = reports_cfg.get("eval_report_path", "reports/eval_report.md")
    report_md = f"""# Model Evaluation Report
- **Model Type**: {model_cfg.get('type')}
- **Hyperparameters**: n_estimators={model_cfg.get('n_estimators')}, max_depth={model_cfg.get('max_depth')}
- **Dataset**: {data_cfg.get('raw_path')} (Train: {len(X_train)}, Test: {len(X_test)})
- **Target**: {data_cfg.get('target_column')}

## Performance Metrics on Held-out Test Set
| Metric | Value |
|---|---|
| **$R^2$ Score** | {r2:.4f} |
| **RMSE** | ${rmse:,.2f} |
| **MAE** | ${mae:,.2f} |

## Pipeline Architecture
- Step 1: `StandardScaler` (feature normalization)
- Step 2: `RandomForestRegressor` (ensemble predictor)
"""
    with open(eval_report_path, "w") as f:
        f.write(report_md)
    print(f"      Saved evaluation report to: {eval_report_path}")

    # Optional MLflow tracking hook
    try:
        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("mlops-house-price-prediction")
        with mlflow.start_run():
            mlflow.log_params(model_cfg)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")
            print("      Successfully logged run to MLflow.")
    except Exception as e:
        # Gracefully proceed if MLflow is not running/installed
        pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train house price prediction pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()

