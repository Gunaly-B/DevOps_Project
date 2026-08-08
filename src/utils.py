"""
Shared Utility Helpers (Logging, Path Handling, Serialization).
"""

import logging
import os
from pathlib import Path
from typing import Any
import joblib


def setup_logger(name: str = "MLOpsPipeline") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def ensure_directory(path: Path) -> Path:
    """Ensures parent directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(obj: Any, file_path: Path) -> None:
    """Saves object using joblib."""
    ensure_directory(file_path.parent)
    joblib.dump(obj, file_path)


def load_artifact(file_path: Path) -> Any:
    """Loads object using joblib."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found at {file_path}")
    return joblib.load(file_path)
