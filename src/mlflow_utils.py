"""
MLflow Utilities for Experiment Tracking and Model Registry Management.
"""

import sys
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import Config, get_config
from src.utils import setup_logger

logger = setup_logger("MLflowUtils")


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking and model registry operations.
    """

    def __init__(self, config: Config = None) -> None:
        self.config = config or get_config()
        self.tracking_uri = self.config.tracking_uri
        self.experiment_name = self.config.experiment_name
        self.registered_model_name = self.config.registered_model_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Any,
        input_example: Optional[Any] = None,
    ) -> str:
        """
        Logs a single training run parameters, metrics, and sklearn model artifact to MLflow.
        Returns the active run_id.
        """
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run '{run_name}' (Run ID: {run_id})")

            # Log hyperparameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
            )

            logger.info(f"Successfully logged run '{run_name}' to MLflow.")
            return run_id

    def register_and_promote_model(
        self, run_id: str, alias: str = "Production", stage: str = "Production"
    ) -> Any:
        """
        Registers the model from the specified run_id in MLflow Model Registry
        and promotes it using both MLflow Alias and Stage transition.
        """
        model_uri = f"runs:/{run_id}/model"
        logger.info(
            f"Registering model '{self.registered_model_name}' from URI {model_uri}..."
        )

        model_version = mlflow.register_model(
            model_uri=model_uri, name=self.registered_model_name
        )

        version_str = str(model_version.version)
        logger.info(f"Registered version {version_str} for model '{self.registered_model_name}'.")

        # Set Alias (Modern MLflow pattern)
        try:
            self.client.set_registered_model_alias(
                name=self.registered_model_name,
                alias=alias,
                version=version_str,
            )
            logger.info(f"Assigned alias '{alias}' to model version {version_str}.")
        except Exception as e:
            logger.warning(f"Could not set alias on registered model: {e}")

        # Transition Stage (Classic MLflow pattern)
        try:
            self.client.transition_model_version_stage(
                name=self.registered_model_name,
                version=version_str,
                stage=stage,
                archive_existing_versions=True,
            )
            logger.info(f"Promoted model version {version_str} to stage '{stage}'.")
        except Exception as e:
            logger.warning(f"Could not transition stage on registered model: {e}")

        return model_version
