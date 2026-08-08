"""
Training Pipeline Orchestrator Script.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from src.config import Config, get_config
from src.data_loader import load_train_test_data
from src.evaluate import evaluate_model, select_best_model
from src.mlflow_utils import MLflowTracker
from src.models import ModelFactory
from src.preprocessing import DataPreprocessor
from src.utils import save_artifact, setup_logger

logger = setup_logger("TrainPipeline")


def run_training_pipeline(config: Config = None) -> Tuple[str, Dict[str, float]]:
    """
    Executes the end-to-end training, MLflow tracking, and model registration pipeline.
    Returns best model name and its evaluation metrics dictionary.
    """
    if config is None:
        config = get_config()

    logger.info("========== Starting MLOps Training Pipeline ==========")

    # Step 1: Load train & test dataset splits
    train_df, test_df = load_train_test_data(config)
    logger.info(f"Loaded train set shape: {train_df.shape}, test set shape: {test_df.shape}")

    # Step 2: Fit preprocessor & transform datasets
    preprocessor = DataPreprocessor(config)
    X_train_trans, y_train = preprocessor.fit_transform(train_df)
    X_test_trans, y_test = preprocessor.transform(test_df)

    # Save fitted preprocessor artifact
    preprocessor.save(config.preprocessor_path)

    # Step 3: Initialize MLflow tracking
    tracker = MLflowTracker(config)

    # Step 4: Candidate model factory loop
    candidate_models = ModelFactory.get_all_candidate_models(config)
    model_results: Dict[str, Dict[str, float]] = {}
    model_instances: Dict[str, Any] = {}
    run_ids: Dict[str, str] = {}

    for model_name, model in candidate_models.items():
        logger.info(f"--- Training Candidate Model: {model_name} ---")
        model.fit(X_train_trans, y_train)

        # Evaluate model performance
        metrics = evaluate_model(model, X_test_trans, y_test)
        model_results[model_name] = metrics
        model_instances[model_name] = model

        # Get hyperparameters dictionary
        params = model.get_params()

        # Input example for MLflow schema enforcement
        input_example = X_train_trans[:2]

        # Log run in MLflow
        run_id = tracker.log_run(
            run_name=f"Run_{model_name}",
            params=params,
            metrics=metrics,
            model=model,
            input_example=input_example,
        )
        run_ids[model_name] = run_id

    # Step 5: Select best model based on configured primary metric
    best_model_name, best_score = select_best_model(
        model_results=model_results, primary_metric=config.primary_metric
    )

    best_run_id = run_ids[best_model_name]
    best_model_instance = model_instances[best_model_name]

    logger.info(
        f"Promoting best model '{best_model_name}' (Run ID: {best_run_id}) "
        f"with {config.primary_metric} = {best_score:.4f}"
    )

    # Step 6: Register & promote best model in MLflow Model Registry
    try:
        tracker.register_and_promote_model(
            run_id=best_run_id,
            alias=config.mlflow_alias,
            stage=config.mlflow_stage,
        )
    except Exception as e:
        logger.warning(f"Could not complete MLflow registry promotion: {e}")

    # Step 7: Save local model artifact fallback for standalone deployment
    save_artifact(best_model_instance, config.model_path)
    logger.info(f"Best model artifact saved locally to {config.model_path}")

    logger.info("========== MLOps Training Pipeline Completed Successfully ==========")
    return best_model_name, model_results[best_model_name]


if __name__ == "__main__":
    run_training_pipeline()
