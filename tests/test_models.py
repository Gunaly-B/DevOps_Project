"""
Unit tests for model factory module.
"""

import pytest
import numpy as np
from src.config import get_config
from src.models import ModelFactory


def test_model_factory_instantiation():
    config = get_config()
    logreg = ModelFactory.get_model("logistic_regression", config)
    rf = ModelFactory.get_model("random_forest", config)
    gb = ModelFactory.get_model("gradient_boosting", config)

    assert logreg is not None
    assert rf is not None
    assert gb is not None


def test_invalid_model_name_raises():
    with pytest.raises(ValueError):
        ModelFactory.get_model("invalid_model_name")


def test_model_fit_predict():
    config = get_config()
    model = ModelFactory.get_model("logistic_regression", config)
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 2, size=20)

    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 20
    assert set(preds).issubset({0, 1})
