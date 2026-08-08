"""
Data Ingestion and Train/Test Splitting Module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import Config, get_config
from src.utils import ensure_directory, setup_logger

logger = setup_logger("DataLoader")

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def ingest_raw_data(config: Config) -> pd.DataFrame:
    """
    Ingests raw dataset from UCI processed.cleveland.data or raw CSV file.
    Saves formatted dataset with headers into raw_data_path.
    """
    raw_path = config.raw_data_path
    source_path = config.source_data_path

    ensure_directory(raw_path.parent)

    if source_path.exists():
        logger.info(f"Reading source data from {source_path}")
        df = pd.read_csv(source_path, header=None, names=COLUMN_NAMES, na_values="?")
    elif raw_path.exists():
        logger.info(f"Reading existing raw data from {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        # Fallback dataset download / generation if source file is missing
        logger.warning(f"Source file {source_path} not found. Attempting online download...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, header=None, names=COLUMN_NAMES, na_values="?")

    # Binary target transformation: target > 0 => 1 (heart disease present), 0 => 0
    if "target" in df.columns:
        df["target"] = df["target"].apply(lambda x: 1 if pd.notna(x) and float(x) > 0 else 0)

    # Save to raw_data_path
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw formatted dataset saved to {raw_path} (Shape: {df.shape})")
    return df


def prepare_and_split_data(config: Config = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits raw dataset into train and test sets and saves them into processed directory.
    """
    if config is None:
        config = get_config()

    df = ingest_raw_data(config)

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[config.target_column],
    )

    ensure_directory(config.processed_dir)
    train_df.to_csv(config.train_data_path, index=False)
    test_df.to_csv(config.test_data_path, index=False)

    logger.info(
        f"Data split completed. Train: {train_df.shape}, Test: {test_df.shape}. "
        f"Saved to {config.train_data_path} and {config.test_data_path}"
    )

    return train_df, test_df


def load_train_test_data(config: Config = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test datasets from processed directory.
    Re-creates them if they do not exist.
    """
    if config is None:
        config = get_config()

    if not config.train_data_path.exists() or not config.test_data_path.exists():
        logger.info("Processed train/test files missing. Running prepare_and_split_data...")
        return prepare_and_split_data(config)

    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)
    return train_df, test_df


if __name__ == "__main__":
    prepare_and_split_data()
