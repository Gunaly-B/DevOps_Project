"""
Data Preprocessing, Imputation, Scaling, and Encoding Module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Tuple as TupleType
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, get_config
from src.utils import load_artifact, save_artifact, setup_logger

logger = setup_logger("Preprocessing")


def build_preprocessor_pipeline(config: Config = None) -> ColumnTransformer:
    """
    Builds Scikit-Learn ColumnTransformer pipeline for feature preprocessing.
    """
    if config is None:
        config = get_config()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, config.numerical_features),
            ("cat", cat_pipeline, config.categorical_features),
        ],
        remainder="passthrough",
    )

    return preprocessor


class DataPreprocessor:
    """
    Data Preprocessor wrapping feature transformations and pipeline persistence.
    """

    def __init__(self, config: Config = None) -> None:
        self.config = config or get_config()
        self.pipeline = build_preprocessor_pipeline(self.config)
        self.is_fitted = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits preprocessor on features X and transforms X, returning (X_transformed, y).
        """
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column].values

        X_trans = self.pipeline.fit_transform(X)
        self.is_fitted = True
        logger.info(f"Preprocessor fitted and transformed X shape: {X_trans.shape}")
        return X_trans, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms features X using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet! Call fit_transform first.")

        if self.config.target_column in df.columns:
            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column].values
        else:
            X = df
            y = None

        X_trans = self.pipeline.transform(X)
        return X_trans, y

    def transform_single_dict(self, input_dict: dict) -> np.ndarray:
        """
        Transforms a single dictionary of input features for inference.
        """
        df = pd.DataFrame([input_dict])
        return self.pipeline.transform(df)

    def save(self, file_path: str = None) -> None:
        """Saves fitted preprocessor to file."""
        target_path = file_path or self.config.preprocessor_path
        save_artifact(self, target_path)
        logger.info(f"Preprocessor artifact saved to {target_path}")

    @classmethod
    def load(cls, file_path: str = None) -> "DataPreprocessor":
        """Loads preprocessor artifact from file."""
        config = get_config()
        target_path = file_path or config.preprocessor_path
        logger.info(f"Loading preprocessor artifact from {target_path}")
        return load_artifact(target_path)
