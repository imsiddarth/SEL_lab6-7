"""
Inference module for batch and standalone predictions.
Loads the serialized Scikit-learn Pipeline and performs scoring.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any


def load_model(model_path: str = "models/model.joblib"):
    """
    Loads saved Scikit-Learn Pipeline from disk.
    
    Args:
        model_path: Path to serialized joblib model.
        
    Returns:
        Loaded Scikit-Learn Pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def predict(model: Any, input_data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> np.ndarray:
    """
    Generates predictions using the loaded pipeline.
    
    Args:
        model: Trained pipeline object.
        input_data: DataFrame, list of feature dictionaries, or single feature dictionary.
        
    Returns:
        np.ndarray containing predicted target values.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

    expected_features = ["square_feet", "bedrooms", "bathrooms", "location_score", "year_built", "garage_spaces"]
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

    # Ensure consistent column ordering
    ordered_df = df[expected_features]
    predictions = model.predict(ordered_df)
    return predictions


if __name__ == "__main__":
    model_file = "models/model.joblib"
    print(f"Loading model pipeline from {model_file}...")
    pipeline = load_model(model_file)

    sample_records = [
        {"square_feet": 1200.0, "bedrooms": 2, "bathrooms": 1.0, "location_score": 6.5, "year_built": 1985, "garage_spaces": 1},
        {"square_feet": 1850.5, "bedrooms": 3, "bathrooms": 2.0, "location_score": 7.2, "year_built": 2002, "garage_spaces": 2},
        {"square_feet": 2400.0, "bedrooms": 4, "bathrooms": 2.5, "location_score": 8.0, "year_built": 2015, "garage_spaces": 2},
        {"square_feet": 3100.0, "bedrooms": 4, "bathrooms": 3.5, "location_score": 9.1, "year_built": 2020, "garage_spaces": 3},
        {"square_feet": 850.0,  "bedrooms": 1, "bathrooms": 1.0, "location_score": 5.0, "year_built": 1970, "garage_spaces": 0},
    ]

    print("\n--- Generating Predictions for 5 Test Records ---")
    preds = predict(pipeline, sample_records)
    for i, (rec, pred) in enumerate(zip(sample_records, preds), start=1):
        print(f"Record {i}: Area={rec['square_feet']}sqft, Beds={rec['bedrooms']}, Score={rec['location_score']} -> Predicted Price: ${pred:,.2f}")

