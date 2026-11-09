"""
FastAPI REST API Service for ML Inference.
Exposes /health, /predict, and /predict_batch endpoints with Pydantic schema validation.
"""
import os
import joblib
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict


# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction ML Service",
    description="Production REST API serving Scikit-Learn Housing Price Prediction Pipeline.",
    version="1.0.0"
)

# Load trained pipeline safely
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model pipeline from: {MODEL_PATH}")
    else:
        model = None
        print(f"Warning: Model file not found at {MODEL_PATH}. API will return 503 until loaded.")
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")


# Pydantic Schemas for Request & Response
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "square_feet": 2100.0,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "location_score": 8.0,
                "year_built": 2010,
                "garage_spaces": 2
            }
        }
    )

    square_feet: float = Field(..., gt=0, le=10000, description="Total interior area in square feet")
    bedrooms: int = Field(..., ge=1, le=15, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=1.0, le=10.0, description="Number of bathrooms")
    location_score: float = Field(..., ge=1.0, le=10.0, description="Neighborhood desirability score (1-10)")
    year_built: int = Field(..., ge=1800, le=2026, description="Construction year")
    garage_spaces: int = Field(..., ge=0, le=10, description="Number of garage parking spaces")


class PredictionOutput(BaseModel):
    prediction: List[float]
    units: str = "USD"
    status: str = "success"
    model_version: str = "1.0.0"


class BatchPredictionOutput(BaseModel):
    predictions: List[float]
    count: int
    units: str = "USD"
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Health check endpoint returning system status and model readiness.
    """
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: PredictionInput):
    """
    Online inference endpoint for individual house price prediction.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )
    
    try:
        # Construct DataFrame with exact feature names and order expected by Scikit-learn Pipeline
        input_df = pd.DataFrame([{
            "square_feet": payload.square_feet,
            "bedrooms": payload.bedrooms,
            "bathrooms": payload.bathrooms,
            "location_score": payload.location_score,
            "year_built": payload.year_built,
            "garage_spaces": payload.garage_spaces
        }])
        
        raw_predictions = model.predict(input_df)
        rounded_predictions = [round(float(val), 2) for val in raw_predictions]
        
        return {
            "prediction": rounded_predictions,
            "units": "USD",
            "status": "success",
            "model_version": "1.0.0"
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating prediction."
        )


@app.post("/predict_batch", response_model=BatchPredictionOutput, tags=["Inference"])
def predict_batch(payload: List[PredictionInput]):
    """
    Batch inference endpoint for scoring multiple records in a single payload.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty."
        )
        
    try:
        records = [item.model_dump() for item in payload]
        input_df = pd.DataFrame(records)[["square_feet", "bedrooms", "bathrooms", "location_score", "year_built", "garage_spaces"]]
        
        raw_predictions = model.predict(input_df)
        rounded_predictions = [round(float(val), 2) for val in raw_predictions]
        
        return {
            "predictions": rounded_predictions,
            "count": len(rounded_predictions),
            "units": "USD",
            "status": "success"
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing batch prediction."
        )
