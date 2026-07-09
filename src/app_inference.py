from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_b_dataset_audit import audit_and_clean
from src.block_d_preprocessing import run_block_d
from src.features import assign_time_band, normalize_clase, normalize_clima, normalize_zona, parse_hour_value
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
def load_calibrator() -> dict[str, Any] | None:
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    if not calibrator_path.exists():
        return None
    return joblib.load(calibrator_path)


def apply_calibration(probabilities: np.ndarray, calibrator: dict[str, Any] | None) -> np.ndarray:
    if calibrator is None:
        return probabilities
    method = calibrator.get("method")
    model = calibrator.get("model")
    if method == "platt":
        return model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return model.predict(probabilities)
    return probabilities


@lru_cache(maxsize=1)
def load_prediction_stack() -> tuple[keras.Model, Any, dict[str, Any], float, dict[str, Any] | None]:
    model = keras.models.load_model(MODELS_DIR / "letalidad_nn.keras")
    scaler, encoders = load_artifacts(MODELS_DIR)
    threshold = load_threshold()
    calibrator = load_calibrator()
    return model, scaler, encoders, threshold, calibrator


def predict_records(records: pd.DataFrame) -> pd.DataFrame:
    model, scaler, encoders, threshold, calibrator = load_prediction_stack()
    features = preparar_entrada(records, scaler=scaler, encoders=encoders)
    probabilities = model.predict(features, verbose=0).reshape(-1)
    calibrated = apply_calibration(probabilities, calibrator)
    output = records.copy()
    output["probabilidad_multifatal"] = probabilities
    output["score_riesgo_multifatal"] = calibrated
    output["calibracion"] = "sin_calibrador" if calibrator is None else str(calibrator.get("method", "desconocida"))
    output["clasificacion"] = np.where(probabilities >= threshold, "ALTA_LETALIDAD", "LETALIDAD_SIMPLE")
    output["threshold"] = threshold
    return output


def load_demo_cases() -> pd.DataFrame:
    ensure_runtime_processed_files()
    return pd.read_csv(PROCESSED_DIR / "demo_cases.csv")


def load_clean_dataset() -> pd.DataFrame:
    ensure_runtime_processed_files()
    return pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet")


def _rate_row(label: str, rate: float | None, support: int | None, source: str) -> dict[str, Any]:
    return {
        "comparador": label,
        "tasa_multifatal": rate,
        "soporte": support,
        "fuente": source,
    }


def observed_multifatal_rate(df: pd.DataFrame, mask: pd.Series) -> tuple[float | None, int]:
    subset = df.loc[mask]
    support = int(subset.shape[0])
    if support == 0:
        return None, support
    return float(subset["target_multifatal"].mean()), support


def historical_comparison(record: pd.Series | dict[str, Any], predicted_probability: float) -> pd.DataFrame:
    """Compare one prediction against observed historical multifatal rates.

    The comparison uses the clean dataset only as observational context. It is
    not a label for the manual input and must not be presented as ground truth
    for a hypothetical crash.
    """

    df = load_clean_dataset().copy()
    df["clase_norm"] = df["CLASE"].apply(normalize_clase)
    df["zona_norm"] = df["ZONA"].apply(normalize_zona)
    df["clima_norm"] = df["CLIMA"].apply(normalize_clima)

    data = dict(record)
    clase = normalize_clase(data.get("CLASE"))
    zona = normalize_zona(data.get("ZONA"))
    clima = normalize_clima(data.get("CLIMA"))
    departamento = str(data.get("DEPARTAMENTO", "")).strip().upper()
    hour = parse_hour_value(data.get("HORA"))
    franja = assign_time_band(hour)

    rows: list[dict[str, Any]] = [
        _rate_row("Score calibrado del modelo", float(predicted_probability), 1, "MLP + calibración post-hoc"),
    ]

    global_rate, global_support = observed_multifatal_rate(df, pd.Series(True, index=df.index))
    rows.append(_rate_row("Histórico global", global_rate, global_support, "Dataset limpio"))

    clase_rate, clase_support = observed_multifatal_rate(df, df["clase_norm"] == clase)
    rows.append(_rate_row(f"Histórico clase: {clase}", clase_rate, clase_support, "Dataset limpio"))

    zona_rate, zona_support = observed_multifatal_rate(df, df["zona_norm"] == zona)
    rows.append(_rate_row(f"Histórico zona: {zona}", zona_rate, zona_support, "Dataset limpio"))

    clima_rate, clima_support = observed_multifatal_rate(df, df["clima_norm"] == clima)
    rows.append(_rate_row(f"Histórico clima: {clima}", clima_rate, clima_support, "Dataset limpio"))

    dept_rate, dept_support = observed_multifatal_rate(df, df["DEPARTAMENTO"] == departamento)
    rows.append(_rate_row(f"Histórico departamento: {departamento}", dept_rate, dept_support, "Dataset limpio"))

    if pd.notna(hour):
        hour_rate, hour_support = observed_multifatal_rate(df, df["hora_entera"] == int(hour))
        rows.append(_rate_row(f"Histórico hora: {int(hour):02d}:00", hour_rate, hour_support, "Dataset limpio"))

    similar_mask = (df["clase_norm"] == clase) & (df["zona_norm"] == zona) & (df["DEPARTAMENTO"] == departamento)
    if pd.notna(hour):
        if franja == "MADRUGADA":
            similar_mask = similar_mask & df["hora_entera"].between(0, 5)
        elif franja == "MANANA":
            similar_mask = similar_mask & df["hora_entera"].between(6, 11)
        elif franja == "TARDE":
            similar_mask = similar_mask & df["hora_entera"].between(12, 17)
        elif franja == "NOCHE":
            similar_mask = similar_mask & df["hora_entera"].between(18, 23)

    similar_rate, similar_support = observed_multifatal_rate(df, similar_mask)
    if similar_support >= 10:
        rows.append(_rate_row("Histórico casos similares", similar_rate, similar_support, "Clase + zona + departamento + franja"))

    comparison = pd.DataFrame(rows)
    comparison["tasa_multifatal_pct"] = comparison["tasa_multifatal"].astype(float) * 100
    return comparison
