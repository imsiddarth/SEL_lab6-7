"""
Unit tests for data processing, pipeline training, and prediction consistency.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_config, load_data, preprocess_data, split_data
from src.predict import load_model, predict


@pytest.fixture
def sample_raw_df():
    """Provides sample DataFrame with valid and edge-case values."""
    return pd.DataFrame({
        "square_feet": [1500.0, 2200.0, -100.0, 1800.0, 1500.0],  # Includes negative and duplicate
        "bedrooms": [3, 4, 2, 3, 3],
        "bathrooms": [2.0, 2.5, 1.0, 2.0, 2.0],
        "location_score": [7.0, 8.5, 5.0, 6.0, 7.0],
        "year_built": [1995, 2010, 1980, 2000, 1995],
        "garage_spaces": [2, 2, 1, 1, 2],
        "price": [550000.0, 780000.0, 300000.0, 620000.0, 550000.0]
    })


def test_data_loading():
    """Test 1: Verify data loading loads non-empty DataFrame with expected columns."""
    config = load_config("configs/config.yaml")
    raw_path = config["data"]["raw_path"]
    df = load_data(raw_path)
    
    assert not df.empty, "Loaded dataframe is empty."
    expected_cols = config["data"]["feature_columns"] + [config["data"]["target_column"]]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' is missing from loaded data."


def test_data_preprocessing_cleans_invalid_records(sample_raw_df):
    """Test 2: Verify preprocessing eliminates duplicates and negative/invalid values."""
    cleaned = preprocess_data(sample_raw_df, target_column="price")
    
    # Negative square feet (-100.0) should be dropped
    assert (cleaned["square_feet"] > 0).all(), "Invalid non-positive square feet found."
    # Duplicate row should be dropped (5 rows -> 3 valid unique rows)
    assert len(cleaned) == 3, f"Expected 3 cleaned rows, got {len(cleaned)}"


def test_data_split_integrity():
    """Test 3: Verify train/test split maintains expected ratio and feature alignment."""
    config = load_config("configs/config.yaml")
    df = load_data(config["data"]["raw_path"])
    target_col = config["data"]["target_column"]
    test_size = config["training"]["test_size"]
    
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size=test_size, random_state=42)
    
    expected_test_len = int(len(df) * test_size)
    assert abs(len(X_test) - expected_test_len) <= 1, "Split test size does not match configuration."
    assert len(X_train) == len(y_train), "Train feature and target counts mismatch."
    assert len(X_test) == len(y_test), "Test feature and target counts mismatch."
    assert target_col not in X_train.columns, "Target column leaked into training features."


def test_prediction_output_shape_and_range():
    """Test 4: Verify prediction output shape, types, and realistic range."""
    model_path = "models/model.joblib"
    assert Path(model_path).exists(), f"Trained model not found at {model_path}"
    model = load_model(model_path)
    
    input_records = [
        {"square_feet": 1200.0, "bedrooms": 2, "bathrooms": 1.0, "location_score": 6.0, "year_built": 1990, "garage_spaces": 1},
        {"square_feet": 2500.0, "bedrooms": 4, "bathrooms": 2.5, "location_score": 8.0, "year_built": 2015, "garage_spaces": 2},
        {"square_feet": 3200.0, "bedrooms": 5, "bathrooms": 3.5, "location_score": 9.0, "year_built": 2022, "garage_spaces": 3},
    ]
    
    predictions = predict(model, input_records)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy ndarray."
    assert len(predictions) == len(input_records), "Prediction count does not match input count."
    assert all(p > 0 for p in predictions), "Predictions must be positive prices."


def test_prediction_missing_features_raises_error():
    """Test 5: Verify input validation raises error when required columns are absent."""
    model = load_model("models/model.joblib")
    invalid_input = [{"square_feet": 1500.0, "bedrooms": 3}]  # Missing required features
    
    with pytest.raises(ValueError, match="missing required feature columns"):
        predict(model, invalid_input)

