from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
TRAINING_DATA_PATH = DATA_DIR / "train.csv"
RAW_DATASET_PATH = DATA_DIR / "dataset.csv"
PREDICTION_INPUT_PATH = DATA_DIR / "predict.csv"
