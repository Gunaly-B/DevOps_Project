"""
Centralized Configuration Loader for MLOps Pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


class Config:
    """Configuration class wrapping settings from config.yaml."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_file = Path(config_path)
        self.raw_config: Dict[str, Any] = self._load_config()

        # Data configurations
        data_cfg = self.raw_config.get("data", {})
        self.raw_data_path = Path(data_cfg.get("raw_path", "data/raw/dataset.csv"))
        self.processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
        self.train_data_path = Path(data_cfg.get("train_path", "data/processed/train.csv"))
        self.test_data_path = Path(data_cfg.get("test_path", "data/processed/test.csv"))
        self.source_data_path = Path(data_cfg.get("source_data", "processed.cleveland.data"))
        self.test_size = float(data_cfg.get("test_size", 0.2))
        self.random_state = int(data_cfg.get("random_state", 42))
        self.target_column = str(data_cfg.get("target_column", "target"))
        self.numerical_features: List[str] = data_cfg.get("numerical_features", [])
        self.categorical_features: List[str] = data_cfg.get("categorical_features", [])

        # Model configurations
        models_cfg = self.raw_config.get("models", {})
        self.primary_metric = str(models_cfg.get("primary_metric", "f1_score"))
        self.logistic_regression_params = models_cfg.get("logistic_regression", {})
        self.random_forest_params = models_cfg.get("random_forest", {})
        self.gradient_boosting_params = models_cfg.get("gradient_boosting", {})

        # MLflow configurations
        mlflow_cfg = self.raw_config.get("mlflow", {})
        self.experiment_name = str(mlflow_cfg.get("experiment_name", "Heart_Disease_Prediction"))
        self.tracking_uri = str(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
        self.registered_model_name = str(mlflow_cfg.get("registered_model_name", "heart-disease-classifier"))
        self.mlflow_alias = str(mlflow_cfg.get("alias", "Production"))
        self.mlflow_stage = str(mlflow_cfg.get("stage", "Production"))

        # Artifacts configurations
        artifacts_cfg = self.raw_config.get("artifacts", {})
        self.models_dir = Path(artifacts_cfg.get("models_dir", "models"))
        self.model_path = Path(artifacts_cfg.get("model_path", "models/best_model.pkl"))
        self.preprocessor_path = Path(artifacts_cfg.get("preprocessor_path", "models/preprocessor.pkl"))

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration file."""
        if not self.config_file.exists():
            return {}
        with open(self.config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


# Convenience instance getter
def get_config(config_path: str = "config.yaml") -> Config:
    return Config(config_path=config_path)
