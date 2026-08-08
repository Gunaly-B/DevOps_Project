"""
FastAPI Prediction Service Application.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, status
from src.predict import HeartDiseasePredictor
from src.schemas import HealthResponse, HeartDiseaseInput, PredictionOutput
from src.utils import setup_logger

logger = setup_logger("FastAPIApp")

app = FastAPI(
    title="Heart Disease Prediction MLOps API",
    description="Production REST API for Heart Disease Risk Prediction using MLflow Registered Models.",
    version="1.0.0",
)

# Global Predictor instance
predictor: HeartDiseasePredictor = None


@app.on_event("startup")
def startup_event():
    """Initializes model predictor on application startup."""
    global predictor
    logger.info("Initializing Heart Disease Predictor on API startup...")
    try:
        predictor = HeartDiseasePredictor()
    except Exception as e:
        logger.error(f"Error initializing predictor during startup: {e}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Readiness and Health Check Endpoint.
    """
    is_ready = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        version="1.0.0",
        model_loaded=is_ready,
    )


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_heart_disease(payload: HeartDiseaseInput):
    """
    Predict Heart Disease risk based on clinical parameters.
    """
    global predictor
    if predictor is None or predictor.model is None:
        # Attempt lazy reload
        try:
            predictor = HeartDiseasePredictor()
        except Exception as e:
            logger.error(f"Failed lazy load of model: {e}")

    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the pipeline first using `python src/train.py`.",
        )

    try:
        result = predictor.predict(payload)
        return result
    except Exception as e:
        logger.error(f"Error executing prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
