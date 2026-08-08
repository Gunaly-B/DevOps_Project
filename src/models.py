"""
Model Factory Module defining candidate classifiers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from src.config import Config, get_config
from src.utils import setup_logger

logger = setup_logger("Models")


class ModelFactory:
    """
    Factory pattern class for creating candidate machine learning classifiers.
    """

    @staticmethod
    def get_model(model_name: str, config: Config = None) -> BaseEstimator:
        """
        Instantiates and returns requested classifier model configured with parameters.
        Supported models: 'logistic_regression', 'random_forest', 'gradient_boosting'
        """
        if config is None:
            config = get_config()

        name = model_name.lower().strip()

        if name in ["logistic_regression", "logreg"]:
            params = config.logistic_regression_params
            logger.info(f"Instantiating LogisticRegression with params: {params}")
            return LogisticRegression(**params)

        elif name in ["random_forest", "rf"]:
            params = config.random_forest_params
            logger.info(f"Instantiating RandomForestClassifier with params: {params}")
            return RandomForestClassifier(**params)

        elif name in ["gradient_boosting", "gb", "xgboost"]:
            params = config.gradient_boosting_params
            logger.info(f"Instantiating GradientBoostingClassifier with params: {params}")
            return GradientBoostingClassifier(**params)

        else:
            raise ValueError(
                f"Unsupported model name '{model_name}'. "
                f"Supported models are: 'logistic_regression', 'random_forest', 'gradient_boosting'."
            )

    @staticmethod
    def get_all_candidate_models(config: Config = None) -> Dict[str, BaseEstimator]:
        """
        Returns a dictionary of all candidate model instances.
        """
        if config is None:
            config = get_config()

        return {
            "logistic_regression": ModelFactory.get_model("logistic_regression", config),
            "random_forest": ModelFactory.get_model("random_forest", config),
            "gradient_boosting": ModelFactory.get_model("gradient_boosting", config),
        }
