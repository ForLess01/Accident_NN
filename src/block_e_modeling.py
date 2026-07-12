"""Train and select the definitive neural network without endpoint leakage.

Architecture, seed and raw decision threshold are selected only with the
2021--2022 training and 2023 selection periods. The separate bundle generator
owns calibration and the read-only 2024--2025 reference evaluation.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_protocol import feature_availability_audit, fit_preprocessor, split_chronological, transform_features

BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
FINAL_MODEL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"
SEEDS = (42, 314, 2718)
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


@dataclass(frozen=True)
class MLPConfig:
    config_id: str
    hidden_units: tuple[int, ...]
    dropout: float
    l2: float
    learning_rate: float
    batch_size: int


# Compact grid declared before looking at selection-period metrics.
MLP_GRID = (
    MLPConfig("MLP_32_16", (32, 16), 0.25, 1e-4, 1e-3, 64),
    MLPConfig("MLP_64_32", (64, 32), 0.35, 3e-4, 5e-4, 64),
    MLPConfig("MLP_32", (32,), 0.20, 1e-4, 5e-4, 64),
)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
    """Common predeclared validation-only decision policy for every model."""
    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
            }
        )
    # Exact order is predeclared: maximize F1; use recall then precision only
    # to resolve a numeric tie, then prefer the higher threshold.
    selected = sorted(rows, key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]), reverse=True)[0]
    selected["selection_policy"] = "max validation F1; ties: recall, precision, higher threshold"
    return selected


def evaluate(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    prediction = (probabilities >= threshold).astype(int)
    return {
        "f1_multifatal": float(f1_score(y, prediction, zero_division=0)),
        "precision_multifatal": float(precision_score(y, prediction, zero_division=0)),
        "recall_multifatal": float(recall_score(y, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & ((probabilities < upper) if upper < 1 else (probabilities <= upper))
        if mask.any():
            value += mask.mean() * abs(float(y_true[mask].mean()) - float(probabilities[mask].mean()))
    return float(value)


def bootstrap_ci(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, iterations: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260709)
    y = np.asarray(y_true, dtype=int)
    samples: dict[str, list[float]] = {key: [] for key in ["f1_multifatal", "precision_multifatal", "recall_multifatal", "pr_auc", "roc_auc"]}
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        sample_y, sample_p = y[indices], probabilities[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        for key, value in evaluate(sample_y, sample_p, threshold).items():
            samples[key].append(value)
    return pd.DataFrame(
        {
            "metric": list(samples),
            "estimate": [evaluate(y, probabilities, threshold)[key] for key in samples],
            "ci_2_5": [float(np.quantile(values, 0.025)) for values in samples.values()],
            "ci_97_5": [float(np.quantile(values, 0.975)) for values in samples.values()],
            "bootstrap_iterations": iterations,
        }
    )


def build_mlp(feature_count: int, config: MLPConfig) -> keras.Model:
    model = keras.Sequential([layers.Input(shape=(feature_count,))])
    for units in config.hidden_units:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(config.l2)))
        model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def class_weights(y_train: pd.Series) -> dict[int, float]:
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def train_mlp_grid(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[keras.Model, MLPConfig, int, dict[str, float | str], np.ndarray, pd.DataFrame, pd.DataFrame]:
    fitted: dict[tuple[str, int], keras.Model] = {}
    records: list[dict[str, float | int | str]] = []
    weights = class_weights(y_train)
    for config in MLP_GRID:
        for seed in SEEDS:
            set_global_seed(seed)
            model = build_mlp(X_train.shape[1], config)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=20, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=8, min_lr=1e-5),
            ]
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=180,
                batch_size=config.batch_size,
                class_weight=weights,
                callbacks=callbacks,
                verbose=0,
            )
            probabilities = model.predict(X_val, verbose=0).reshape(-1)
            threshold = choose_threshold(y_val, probabilities)
            metrics = evaluate(y_val, probabilities, float(threshold["threshold"]))
            records.append(
                {
                    "config_id": config.config_id,
                    "seed": seed,
                    "hidden_units": "-".join(map(str, config.hidden_units)),
                    "dropout": config.dropout,
                    "l2": config.l2,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "epochs_ran": len(history.history["loss"]),
                    "best_val_pr_auc_history": float(max(history.history["val_pr_auc"])),
                    "selected_threshold": float(threshold["threshold"]),
                    **metrics,
                }
            )
            fitted[(config.config_id, seed)] = model

    run_table = pd.DataFrame(records)
    aggregate = (
        run_table.groupby("config_id", as_index=False)
        .agg(median_val_pr_auc=("pr_auc", "median"), median_val_f1=("f1_multifatal", "median"), pr_auc_iqr=("pr_auc", lambda x: float(x.quantile(0.75) - x.quantile(0.25))))
        .sort_values(["median_val_pr_auc", "median_val_f1", "pr_auc_iqr", "config_id"], ascending=[False, False, True, True])
    )
    chosen_id = str(aggregate.iloc[0]["config_id"])
    candidate = run_table[run_table["config_id"] == chosen_id].copy()
    target_pr_auc = float(candidate["pr_auc"].median())
    target_f1 = float(candidate["f1_multifatal"].median())
    medoid = candidate.assign(_pr_distance=(candidate["pr_auc"] - target_pr_auc).abs(), _f1_distance=(candidate["f1_multifatal"] - target_f1).abs()).sort_values(["_pr_distance", "_f1_distance", "seed"]).iloc[0]
    selected_config = next(config for config in MLP_GRID if config.config_id == chosen_id)
    selected_seed = int(medoid["seed"])
    selected_model = fitted[(chosen_id, selected_seed)]
    selected_probabilities = selected_model.predict(X_val, verbose=0).reshape(-1)
    selected_threshold = choose_threshold(y_val, selected_probabilities)
    return selected_model, selected_config, selected_seed, selected_threshold, selected_probabilities, run_table, aggregate


def train_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict[str, object], pd.DataFrame]:
    models: dict[str, object] = {
        "DummyClassifier_prior": DummyClassifier(strategy="prior", random_state=42),
        "LogisticRegression_balanced": LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42),
        "RandomForest_balanced": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    }
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)  # type: ignore[attr-defined]
        probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]  # type: ignore[attr-defined]
        threshold = choose_threshold(y_val, probabilities)
        rows.append({"model": name, **threshold, **evaluate(y_val, probabilities, float(threshold["threshold"]))})
    return models, pd.DataFrame(rows)


def _platt_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def write_protocol_document(feature_count: int, split_sizes: dict[str, int]) -> None:
    text = f"""# Protocolo definitivo del modelo

## Protocolo predeclarado
- Entrenamiento: siniestros de 2021--2022 ({split_sizes['train']} registros).
- Selección: 2023 ({split_sizes['validation']} registros).
- Referencia histórica: 2024--2025 ({split_sizes['reference']} registros; 2025 es parcial en esta extracción). Este periodo no participa en entrenamiento, selección ni calibración.
- MLP: 3 arquitecturas predeclaradas × 3 semillas = 9 corridas; L2, dropout y early stopping por PR-AUC de validación.
- Selección de arquitectura: mayor mediana de PR-AUC entre semillas; desempates por mediana de F1 e IQR menor. Se conserva la semilla mediana, no la corrida extrema.
- Umbral: la misma regla para MLP, regresión logística y bosque aleatorio: máximo F1 de validación en la grilla 0.05--0.95; desempates por recall, precision y umbral alto.

## Contrato de variables
La matriz tiene {feature_count} columnas y excluye resultados, conteos posteriores, causas investigadas, identificadores y señales con faltantes sesgados. Las variables de clase de siniestro son válidas solo para priorización posterior a la notificación; no se presentan como predicción preventiva previa al evento.

## Límites
El objetivo es multifatalidad condicional a siniestros ya fatales, no la probabilidad de que un siniestro cualquiera sea mortal. La evaluación cronológica mide generalización futura interna, pero no reemplaza validación externa. La calibración se selecciona y ajusta exclusivamente en 2023; el periodo de referencia se usa solo para describir desempeño histórico.
"""
    (TABLES_DIR / "final_model_protocol.md").write_text(text, encoding="utf-8")


def run_modeling_pipeline() -> dict[str, object]:
    """Train and persist the selected MLP without reading the reference period."""
    for directory in [FINAL_MODEL_DIR, TABLES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PATH)
    splits = split_chronological(base)
    X_train_raw, y_train = splits["X_train_raw"], splits["y_train"]
    X_val_raw, y_val = splits["X_validation_raw"], splits["y_validation"]
    # Do not transform or score the reference partition during model selection.
    scaler, encoders = fit_preprocessor(X_train_raw)  # type: ignore[arg-type]
    X_train = transform_features(X_train_raw, scaler, encoders)  # type: ignore[arg-type]
    X_val = transform_features(X_val_raw, scaler, encoders)  # type: ignore[arg-type]
    joblib.dump(scaler, FINAL_MODEL_DIR / "scaler.joblib")
    joblib.dump(encoders, FINAL_MODEL_DIR / "encoders.joblib")
    (FINAL_MODEL_DIR / "feature_list.json").write_text(json.dumps(encoders["feature_list"], ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_models, baseline_validation = train_baselines(X_train, y_train, X_val, y_val)  # type: ignore[arg-type]
    mlp, config, seed, mlp_threshold, mlp_validation_probabilities, grid_table, aggregate_table = train_mlp_grid(X_train, y_train, X_val, y_val)  # type: ignore[arg-type]
    selection = {
        "reference_period_used_during_selection": False,
        "selected_config": asdict(config),
        "selected_seed": seed,
        "selected_threshold": mlp_threshold,
        "selection_rule": "median validation PR-AUC across seeds, then median F1, then lower PR-AUC IQR",
        "threshold_policy": mlp_threshold["selection_policy"],
    }
    baseline_validation.to_csv(TABLES_DIR / "model_selection_baseline_validation.csv", index=False)
    grid_table.to_csv(TABLES_DIR / "model_selection_seed_grid_validation.csv", index=False)
    aggregate_table.to_csv(TABLES_DIR / "model_selection_robustness.csv", index=False)
    (FINAL_MODEL_DIR / "model_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    mlp.save(FINAL_MODEL_DIR / "model.keras")

    feature_availability_audit().to_csv(TABLES_DIR / "feature_availability_audit.csv", index=False)
    split_sizes = {"train": len(y_train), "validation": len(y_val), "reference": len(splits["y_test"])}
    write_protocol_document(len(encoders["feature_list"]), split_sizes)
    summary = {
        "split_sizes": split_sizes,
        "feature_count": len(encoders["feature_list"]),
        "selected_config": config.config_id,
        "selected_seed": seed,
        "selected_threshold": float(mlp_threshold["threshold"]),
        "reference_period_used_for_selection": False,
    }
    (TABLES_DIR / "model_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_modeling_pipeline(), ensure_ascii=False, indent=2))
