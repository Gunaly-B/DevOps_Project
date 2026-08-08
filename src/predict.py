"""
Inference Module loading Registered Model from MLflow Model Registry / Local Artifact.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from src.config import Config, get_config
from src.preprocessing import DataPreprocessor
from src.schemas import HeartDiseaseInput, PredictionOutput
from src.utils import load_artifact, setup_logger

logger = setup_logger("Predictor")


class HeartDiseasePredictor:
    """
    Predictor class managing model loading and inference execution.
    """

    def __init__(self, config: Config = None) -> None:
        self.config = config or get_config()
        self.model = None
        self.preprocessor = None
        self.model_version_info = "Local Fallback Artifact"
        self.load_model_and_preprocessor()

    def load_model_and_preprocessor(self) -> None:
        """
        Attempts to load model from MLflow Model Registry first.
        Falls back to local saved artifact if MLflow registry is unavailable.
        """
        # Step 1: Load preprocessor
        try:
            self.preprocessor = DataPreprocessor.load(self.config.preprocessor_path)
            logger.info("Loaded preprocessor successfully.")
        except Exception as e:
            logger.warning(f"Could not load preprocessor from artifact path: {e}. Building new preprocessor instance.")
            self.preprocessor = DataPreprocessor(self.config)

        # Step 2: Try MLflow Model Registry
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Try loading by Production Alias first
        try:
            model_uri = f"models:/{self.config.registered_model_name}@{self.config.mlflow_alias}"
            logger.info(f"Attempting to load model from MLflow Registry URI: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_version_info = f"MLflow Registry Alias: {self.config.mlflow_alias}"
            logger.info(f"Loaded model successfully from MLflow URI: {model_uri}")
            return
        except Exception as e:
            logger.info(f"Could not load model via alias '{self.config.mlflow_alias}': {e}")

        # Try loading by Production Stage next
        try:
            model_uri = f"models:/{self.config.registered_model_name}/{self.config.mlflow_stage}"
            logger.info(f"Attempting to load model from MLflow Registry URI: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_version_info = f"MLflow Registry Stage: {self.config.mlflow_stage}"
            logger.info(f"Loaded model successfully from MLflow Stage URI: {model_uri}")
            return
        except Exception as e:
            logger.info(f"Could not load model via stage '{self.config.mlflow_stage}': {e}")

        # Fallback to local model artifact file
        try:
            logger.info(f"Attempting local artifact fallback: {self.config.model_path}")
            self.model = load_artifact(self.config.model_path)
            self.model_version_info = f"Local Artifact ({self.config.model_path.name})"
            logger.info(f"Loaded model successfully from local artifact: {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from local artifact: {e}")
            self.model = None

    def predict(self, input_data: HeartDiseaseInput) -> PredictionOutput:
        """
        Executes model prediction on validated Pydantic input features.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure model artifact or MLflow registry exists.")

        # Convert Pydantic payload to DataFrame row
        df_input = pd.DataFrame([input_data.to_dict()])

        # Preprocess features
        X_trans, _ = self.preprocessor.transform(df_input)

        # Execute prediction
        prediction_val = int(self.model.predict(X_trans)[0])

        # Get probability if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            prob_matrix = self.model.predict_proba(X_trans)
            if prob_matrix.ndim == 2 and prob_matrix.shape[1] == 2:
                probability_val = float(prob_matrix[0, 1])
            else:
                probability_val = float(prob_matrix[0, 0])
        else:
            probability_val = float(prediction_val)

        risk_str = "High Risk (Heart Disease Present)" if prediction_val == 1 else "Low Risk (Heart Disease Absent)"

        return PredictionOutput(
            prediction=prediction_val,
            heart_disease_risk=risk_str,
            probability=round(probability_val, 4),
            model_version=self.model_version_info,
        )
