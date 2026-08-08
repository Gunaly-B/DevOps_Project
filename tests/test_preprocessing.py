"""
Unit tests for data preprocessing module.
"""

import pandas as pd
import numpy as np
import pytest

from src.config import get_config
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_dataframe():
    data = {
        "age": [63.0, 67.0, 67.0, 37.0, 41.0],
        "sex": [1.0, 1.0, 1.0, 1.0, 0.0],
        "cp": [1.0, 4.0, 4.0, 3.0, 2.0],
        "trestbps": [145.0, 160.0, 120.0, 130.0, 130.0],
        "chol": [233.0, 286.0, 229.0, 250.0, 204.0],
        "fbs": [1.0, 0.0, 0.0, 0.0, 0.0],
        "restecg": [2.0, 2.0, 2.0, 0.0, 2.0],
        "thalach": [150.0, 108.0, 129.0, 187.0, 172.0],
        "exang": [0.0, 1.0, 1.0, 0.0, 0.0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
        "slope": [3.0, 2.0, 2.0, 3.0, 1.0],
        "ca": [0.0, 3.0, 2.0, 0.0, 0.0],
        "thal": [6.0, 3.0, 7.0, 3.0, 3.0],
        "target": [0, 1, 1, 0, 0],
    }
    return pd.DataFrame(data)


def test_fit_transform(sample_dataframe):
    config = get_config()
    preprocessor = DataPreprocessor(config)
    X_trans, y = preprocessor.fit_transform(sample_dataframe)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(sample_dataframe)
    assert len(y) == len(sample_dataframe)
    assert preprocessor.is_fitted is True


def test_transform_unfitted_raises_error(sample_dataframe):
    config = get_config()
    preprocessor = DataPreprocessor(config)
    with pytest.raises(ValueError):
        preprocessor.transform(sample_dataframe)
