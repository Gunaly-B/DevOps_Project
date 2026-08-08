"""
Model Evaluation and Performance Comparison Module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import setup_logger

logger = setup_logger("Evaluate")


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculates classification performance metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    if y_prob is not None:
        try:
            # Handle binary classification probabilities (column index 1)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                prob_vec = y_prob[:, 1]
            else:
                prob_vec = y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_vec))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC score: {e}")
            metrics["roc_auc"] = 0.0

    return metrics


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates trained model on test feature matrix X_test and ground truth y_test.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    logger.info(f"Model Evaluation Metrics: {metrics}")
    return metrics


def select_best_model(
    model_results: Dict[str, Dict[str, float]], primary_metric: str = "f1_score"
) -> Tuple[str, float]:
    """
    Compares candidate model metrics and selects the name of the best performing model.
    """
    best_model_name = ""
    best_score = -1.0

    for model_name, metrics in model_results.items():
        score = metrics.get(primary_metric, 0.0)
        logger.info(f"Model '{model_name}' - {primary_metric}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model_name = model_name

    logger.info(f"Selected best model '{best_model_name}' with {primary_metric} = {best_score:.4f}")
    return best_model_name, best_score
