"""FastAPI application for credit risk model serving"""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data_processing import FEATURE_SCHEMA_PATH, WOE_MAPPING_PATH, PREPROCESSOR_PATH

from .pydantic_models import (
    CustomerFeatures,
    HealthCheckResponse,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and metadata
model = None
preprocessor = None
feature_columns: List[str] = []
woe_mappings: Dict[str, Dict[str, float]] = {}
mlflow_client: Optional[MlflowClient] = None

TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'credit-risk-model-best-model')
MODEL_STAGE = os.getenv('MLFLOW_MODEL_STAGE', 'Production')
CUSTOM_MODEL_URI = os.getenv('MLFLOW_MODEL_URI')
CATEGORICAL_INPUTS = ['primary_channel', 'primary_category', 'primary_currency', 'primary_pricing']


def build_model_uri() -> str:
    """Compose model URI from env, local path, or registry defaults."""

    if CUSTOM_MODEL_URI:
        return CUSTOM_MODEL_URI
    
    # Check for local model first (preferred for container deployment)
    local_model_path = Path("models") / MODEL_NAME
    if local_model_path.exists():
        logger.info(f"Found local model at {local_model_path}")
        return str(local_model_path)

    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"


def load_feature_metadata(run_id: Optional[str] = None) -> None:
    """Load feature schema, WoE mappings, and preprocessor from local artifacts or registry run artifacts."""

    global feature_columns, woe_mappings, preprocessor

    def _safe_read(path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text())
        return {}

    feature_schema = _safe_read(FEATURE_SCHEMA_PATH)
    woe_mapping_data = _safe_read(WOE_MAPPING_PATH)
    
    if PREPROCESSOR_PATH.exists():
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info("Loaded preprocessor from local path")
        except Exception as e:
            logger.warning("Failed to load local preprocessor: %s", e)

    if (not feature_schema or not woe_mapping_data or preprocessor is None) and run_id and mlflow_client:
        tmp_dir = Path(tempfile.mkdtemp(prefix="metadata_"))
        try:
            fs_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="metadata/feature_schema.json",
                dst_path=tmp_dir,
            )
            wm_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="metadata/woe_mappings.json",
                dst_path=tmp_dir,
            )
            # Try to download preprocessor if not found locally
            if preprocessor is None:
                pp_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="metadata/feature_preprocessor.joblib",
                    dst_path=tmp_dir,
                )
                preprocessor = joblib.load(pp_path)
                
            feature_schema = json.loads(Path(fs_path).read_text())
            woe_mapping_data = json.loads(Path(wm_path).read_text())
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Could not download feature metadata from MLflow: %s", exc)

    feature_columns = feature_schema.get('features', [])
    if not feature_columns:
        raise ValueError("Feature schema is empty")

    woe_mappings = woe_mapping_data
    logger.info("Loaded %s feature columns and %s WoE mappings", len(feature_columns), len(woe_mappings))
    if preprocessor:
        logger.info("Preprocessor loaded successfully")
    else:
        logger.warning("Preprocessor not loaded - predictions may be inaccurate")


def build_feature_frame(features: CustomerFeatures):
    """Transform incoming payload into model-ready dataframe"""

    if not feature_columns:
        raise HTTPException(status_code=503, detail="Feature metadata not loaded.")

    payload = features.dict()
    customer_id = payload.pop('customer_id')
    frame = pd.DataFrame([payload])

    for cat_col in CATEGORICAL_INPUTS:
        mapping = woe_mappings.get(cat_col, {})
        woe_col = f"{cat_col}_woe"
        frame[cat_col] = frame[cat_col].astype(str)
        frame[woe_col] = frame[cat_col].map(mapping).fillna(0.0)

    frame = frame.drop(columns=CATEGORICAL_INPUTS, errors='ignore')
    frame = frame.reindex(columns=feature_columns, fill_value=0.0)
    
    if preprocessor:
        # Apply scaling/imputation using the loaded preprocessor
        # The preprocessor expects a DataFrame with the correct columns
        transformed_array = preprocessor.transform(frame)
        # Convert back to DataFrame to keep feature names for the model
        frame = pd.DataFrame(transformed_array, columns=feature_columns)
        
    return customer_id, frame


def _resolve_registry_run_id() -> Optional[str]:
    if not mlflow_client:
        return None
    try:
        versions = mlflow_client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            return versions[0].run_id
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not resolve registry run id: %s", exc)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""

    global model, mlflow_client

    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
        logger.info("Using MLflow tracking URI: %s", TRACKING_URI)
    mlflow_client = MlflowClient()

    registry_run_id = _resolve_registry_run_id()

    try:
        model_uri = build_model_uri()
        logger.info("Loading model from %s", model_uri)
        
        # Try loading directly with joblib if it's a local directory with model.pkl
        # This avoids MLflow version compatibility issues
        local_pkl_path = Path(model_uri) / "model.pkl"
        if Path(model_uri).exists() and local_pkl_path.exists():
            logger.info("Loading model directly from pickle: %s", local_pkl_path)
            model = joblib.load(local_pkl_path)
        else:
            model = mlflow.sklearn.load_model(model_uri)
            
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.warning("Could not load model: %s", exc)
        model = None

    try:
        load_feature_metadata(registry_run_id)
    except Exception as exc:
        logger.warning("Could not load feature metadata: %s", exc)
    
    yield

    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting credit risk probability and loan recommendations",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Scoring API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    status = "healthy" if model is not None else "degraded"
    model_version = "production" if model is not None else "none"
    
    return HealthCheckResponse(
        status=status,
        model_version=model_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: CustomerFeatures):
    """
    Predict credit risk and loan recommendations for a customer
    
    Example request:
    ```json
    {
        "customer_id": "CUST_12345",
        "total_transaction_amount": 5000.0,
        "transaction_count": 25,
        "avg_transaction_amount": 200.0,
        "recency_days": 5,
        "frequency_score": 0.85,
        "monetary_score": 0.72
    }
    ```
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    try:
        customer_id, X = build_feature_frame(features)
        
        # Make predictions
        risk_prob = model.predict_proba(X)[0, 1]
        risk_category = "high_risk" if risk_prob >= 0.5 else "low_risk"
        
        # Credit score (300-850)
        credit_score = int(300 + (1 - risk_prob) * 550)
        
        # Loan recommendations
        recommended_amount = int(10000 + (1 - risk_prob) * 90000)
        recommended_duration = int(6 + (1 - risk_prob) * 30)
        
        return PredictionResponse(
            customer_id=customer_id,
            risk_probability=float(risk_prob),
            risk_category=risk_category,
            credit_score=credit_score,
            recommended_amount=float(recommended_amount),
            recommended_duration_months=recommended_duration,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(features_list: list[CustomerFeatures]):
    """
    Predict credit risk for multiple customers
    """
    results = []
    for features in features_list:
        try:
            result = await predict(features)
            results.append(result)
        except HTTPException as e:
            logger.error(f"Error predicting for {features.customer_id}: {e.detail}")
    
    return {"predictions": results, "count": len(results)}

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "model_uri": build_model_uri(),
        "feature_count": len(feature_columns),
        "categorical_inputs": CATEGORICAL_INPUTS,
        "available_methods": [m for m in dir(model) if not m.startswith('_')]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
