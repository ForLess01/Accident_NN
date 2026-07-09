from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_b_dataset_audit import audit_and_clean
from src.block_d_preprocessing import run_block_d
from src.preprocessing import load_artifacts, preparar_entrada


MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"


def ensure_runtime_processed_files() -> None:
    if not (PROCESSED_DIR / "base_limpia.parquet").exists():
        audit_and_clean()
    required = [
        PROCESSED_DIR / "demo_cases.csv",
        PROCESSED_DIR / "X_train.parquet",
        PROCESSED_DIR / "X_val.parquet",
        PROCESSED_DIR / "X_test.parquet",
    ]
    if any(not path.exists() for path in required):
        run_block_d()


def load_threshold() -> float:
    threshold_data = json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))
    return float(threshold_data["threshold"])


@lru_cache(maxsize=1)
def load_prediction_stack() -> tuple[keras.Model, Any, dict[str, Any], float]:
    model = keras.models.load_model(MODELS_DIR / "severidad_nn.keras")
    scaler, encoders = load_artifacts(MODELS_DIR)
    threshold = load_threshold()
    return model, scaler, encoders, threshold


def predict_records(records: pd.DataFrame) -> pd.DataFrame:
    model, scaler, encoders, threshold = load_prediction_stack()
    features = preparar_entrada(records, scaler=scaler, encoders=encoders)
    probabilities = model.predict(features, verbose=0).reshape(-1)
    output = records.copy()
    output["probabilidad_mortal"] = probabilities
    output["clasificacion"] = np.where(probabilities >= threshold, "MORTAL", "NO_MORTAL")
    output["threshold"] = threshold
    return output


def load_demo_cases() -> pd.DataFrame:
    ensure_runtime_processed_files()
    return pd.read_csv(PROCESSED_DIR / "demo_cases.csv")


def load_clean_dataset() -> pd.DataFrame:
    ensure_runtime_processed_files()
    return pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet")
