"""Validation-only capacity audit: width, depth, epoch budget and sample size.

The canonical grid compared only three configurations. This audit answers, with
measurement rather than assertion, whether that grid was too small: whether more
neurons or more layers would help, whether the fixed epoch budget under-trains
the refit, and whether the model is limited by capacity or by sample size.

Architecture search uses the legitimate selection surface --- fit on 2021, select
on 2022 --- with the predeclared rule (median PR-AUC across seeds). The epoch and
sample-size curves are diagnostics read on the 2023 partition and are explicitly
NOT a re-selection: adopting any change would require a fresh holdout, because
2023 already fixed calibration and thresholds.

The 2024--2025 reference is never opened. The canonical bundle is not modified.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_e_modeling import class_weights, set_global_seed
from src.model_protocol import EXCLUDED_COLUMNS, transform_features

BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
FINAL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"

SEEDS = (42, 314, 2718)
DROPOUT, L2, LEARNING_RATE, BATCH_SIZE = 0.35, 3e-4, 5e-4, 64
MAX_EPOCHS, PATIENCE = 180, 20
CANONICAL = (64, 32)
EPOCH_SWEEP_LIMIT = 70
SAMPLE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)

# Depth 1 to 3, widths from a quarter to four times the canonical first layer.
ARCHITECTURE_GRID = (
    (32,), (64,), (128,),
    (32, 16), (64, 32), (128, 64), (256, 128),
    (64, 32, 16), (128, 64, 32),
)


def build(width: int, units: tuple[int, ...]) -> keras.Model:
    model = keras.Sequential([layers.Input((width,))])
    for size in units:
        model.add(layers.Dense(size, activation="relu", kernel_regularizer=regularizers.l2(L2)))
        model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def load_partitions() -> dict[str, Any]:
    base = pd.read_parquet(BASE_PATH, filters=[("FECHA", "<", datetime(2024, 1, 1))])
    base["FECHA"] = pd.to_datetime(base["FECHA"], errors="coerce")
    if int(base["FECHA"].dt.year.max()) > 2023:
        raise RuntimeError("Capacity audit crossed the frozen 2023 boundary.")
    years = base["FECHA"].dt.year
    scaler, encoders = joblib.load(FINAL_DIR / "scaler.joblib"), joblib.load(FINAL_DIR / "encoders.joblib")

    def block(mask: pd.Series) -> tuple[np.ndarray, pd.Series]:
        partition = base[mask]
        features = partition.drop(columns=[c for c in EXCLUDED_COLUMNS if c in partition])
        X = transform_features(features, scaler, encoders).to_numpy("float32")
        y = partition["target_multifatal"].astype(int).reset_index(drop=True)
        return X, y

    return {
        "fit": block(years.eq(2021)),
        "select": block(years.eq(2022)),
        "train": block(years.isin([2021, 2022])),
        "val": block(years.eq(2023)),
    }


def architecture_search(parts: dict[str, Any]) -> pd.DataFrame:
    """Expanded grid on the legitimate selection surface: fit 2021, select 2022."""
    (X_fit, y_fit), (X_sel, y_sel) = parts["fit"], parts["select"]
    rows: list[dict[str, Any]] = []
    for units in ARCHITECTURE_GRID:
        for seed in SEEDS:
            keras.backend.clear_session()
            set_global_seed(seed)
            model = build(X_fit.shape[1], units)
            history = model.fit(
                X_fit, y_fit, validation_data=(X_sel, y_sel),
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights(y_fit),
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                  patience=PATIENCE, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                                      factor=0.5, patience=8, min_lr=1e-5),
                ],
                verbose=0,
            )
            probabilities = model.predict(X_sel, verbose=0).reshape(-1)
            rows.append({
                "arquitectura": "-".join(map(str, units)),
                "capas_ocultas": len(units),
                "neuronas_totales": sum(units),
                "parametros": int(model.count_params()),
                "seed": seed,
                "mejor_epoca": int(np.argmax(history.history["val_pr_auc"]) + 1),
                "epocas_corridas": len(history.history["loss"]),
                "pr_auc_seleccion_2022": float(average_precision_score(y_sel, probabilities)),
            })
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["arquitectura", "capas_ocultas", "neuronas_totales", "parametros"], as_index=False)
        .agg(
            pr_auc_mediana=("pr_auc_seleccion_2022", "median"),
            pr_auc_iqr=("pr_auc_seleccion_2022", lambda v: float(v.quantile(0.75) - v.quantile(0.25))),
            epoca_mediana=("mejor_epoca", "median"),
        )
        .sort_values("pr_auc_mediana", ascending=False)
    )
    return runs, summary


class ValidationCurve(keras.callbacks.Callback):
    """Record PR-AUC on a held partition after each epoch (diagnostic only)."""

    def __init__(self, X_train, y_train, X_val, y_val) -> None:
        super().__init__()
        self.data = (X_train, y_train, X_val, y_val)
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        X_train, y_train, X_val, y_val = self.data
        train_p = self.model.predict(X_train, verbose=0).reshape(-1)
        val_p = self.model.predict(X_val, verbose=0).reshape(-1)
        self.rows.append({
            "epoca": epoch + 1,
            "pr_auc_entrenamiento_2021_2022": float(average_precision_score(y_train, train_p)),
            "pr_auc_validacion_2023": float(average_precision_score(y_val, val_p)),
        })


def epoch_curve(parts: dict[str, Any]) -> pd.DataFrame:
    """Diagnostic: how the refit behaves beyond the fixed 14-epoch budget."""
    (X_train, y_train), (X_val, y_val) = parts["train"], parts["val"]
    frames = []
    for seed in SEEDS:
        keras.backend.clear_session()
        set_global_seed(seed)
        model = build(X_train.shape[1], CANONICAL)
        recorder = ValidationCurve(X_train, y_train, X_val, y_val)
        model.fit(X_train, y_train, epochs=EPOCH_SWEEP_LIMIT, batch_size=BATCH_SIZE,
                  class_weight=class_weights(y_train), callbacks=[recorder], verbose=0)
        frames.append(pd.DataFrame(recorder.rows).assign(seed=seed))
    curve = pd.concat(frames, ignore_index=True)
    return (
        curve.groupby("epoca", as_index=False)
        .agg(
            train_mediana=("pr_auc_entrenamiento_2021_2022", "median"),
            val_mediana=("pr_auc_validacion_2023", "median"),
            val_min=("pr_auc_validacion_2023", "min"),
            val_max=("pr_auc_validacion_2023", "max"),
        )
    )


def sample_size_curve(parts: dict[str, Any]) -> pd.DataFrame:
    """Diagnostic: is the model limited by sample size or by the variables?"""
    (X_train, y_train), (X_val, y_val) = parts["train"], parts["val"]
    rows: list[dict[str, Any]] = []
    for fraction in SAMPLE_FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            size = int(len(y_train) * fraction)
            index = rng.choice(len(y_train), size=size, replace=False)
            X_subset, y_subset = X_train[index], y_train.iloc[index].reset_index(drop=True)
            if y_subset.nunique() < 2:
                continue
            keras.backend.clear_session()
            set_global_seed(seed)
            model = build(X_train.shape[1], CANONICAL)
            model.fit(X_subset, y_subset, epochs=14, batch_size=BATCH_SIZE,
                      class_weight=class_weights(y_subset), verbose=0)
            probabilities = model.predict(X_val, verbose=0).reshape(-1)
            rows.append({
                "fraccion": fraction,
                "n_entrenamiento": size,
                "positivos": int(y_subset.sum()),
                "seed": seed,
                "pr_auc_validacion_2023": float(average_precision_score(y_val, probabilities)),
            })
    runs = pd.DataFrame(rows)
    return runs, (
        runs.groupby(["fraccion", "n_entrenamiento"], as_index=False)
        .agg(pr_auc_mediana=("pr_auc_validacion_2023", "median"),
             pr_auc_min=("pr_auc_validacion_2023", "min"),
             pr_auc_max=("pr_auc_validacion_2023", "max"))
    )


def figure(arch: pd.DataFrame, epochs: pd.DataFrame, sample: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    base_color, accent, grey = "#496B7C", "#A63F12", "#7A8B94"

    ordered = arch.sort_values("parametros")
    colors = [accent if a == "64-32" else base_color for a in ordered["arquitectura"]]
    axes[0].errorbar(ordered["parametros"], ordered["pr_auc_mediana"],
                     yerr=ordered["pr_auc_iqr"] / 2, fmt="none", ecolor=grey, capsize=3, zorder=1)
    axes[0].scatter(ordered["parametros"], ordered["pr_auc_mediana"], c=colors, s=55, zorder=2)
    for position, (_, row) in enumerate(ordered.iterrows()):
        offset = (0, 13) if position % 2 == 0 else (0, -17)
        axes[0].annotate(row["arquitectura"], (row["parametros"], row["pr_auc_mediana"]),
                         textcoords="offset points", xytext=offset, ha="center", fontsize=7.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parámetros entrenables (escala log)")
    axes[0].set_ylabel("PR-AUC mediana en selección 2022")
    axes[0].set_title("Capacidad frente a desempeño\n(barra = IQR entre semillas)")

    axes[1].plot(epochs["epoca"], epochs["train_mediana"], color=accent, label="Entrenamiento 2021–2022")
    axes[1].plot(epochs["epoca"], epochs["val_mediana"], color=base_color, label="Validación 2023")
    axes[1].fill_between(epochs["epoca"], epochs["val_min"], epochs["val_max"],
                         color=base_color, alpha=0.18)
    axes[1].axvline(14, color=grey, linestyle="--", linewidth=1.2)
    best_epoch = int(epochs.loc[epochs["val_mediana"].idxmax(), "epoca"])
    axes[1].axvline(best_epoch, color=accent, linestyle=":", linewidth=1.2)
    axes[1].annotate(f"presupuesto canónico: 14\nmáximo observado: {best_epoch}",
                     xy=(14, float(epochs.loc[epochs["epoca"].eq(14), "val_mediana"].iloc[0])),
                     xytext=(30, 0.20), fontsize=8, color="#334155",
                     arrowprops=dict(arrowstyle="->", color=grey))
    axes[1].set_xlabel("Época del reajuste")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Presupuesto de épocas\n(banda = rango entre semillas)")
    axes[1].legend(fontsize=8, loc="center right")

    axes[2].plot(sample["n_entrenamiento"], sample["pr_auc_mediana"],
                 marker="o", color=base_color)
    axes[2].fill_between(sample["n_entrenamiento"], sample["pr_auc_min"], sample["pr_auc_max"],
                         color=base_color, alpha=0.18)
    axes[2].set_xlabel("Registros de entrenamiento")
    axes[2].set_ylabel("PR-AUC en validación 2023")
    axes[2].set_title("Curva de aprendizaje frente al tamaño de muestra")

    fig.suptitle("Auditoría de capacidad: ancho, profundidad, épocas y tamaño de muestra",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "Búsqueda de arquitectura en la superficie legítima de selección (ajuste 2021, selección 2022). "
             "Las curvas de época y muestra son diagnósticos sobre 2023, no una nueva selección.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(FIGURES_DIR / "design_capacity_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, Any]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_partitions()

    arch_runs, arch_summary = architecture_search(parts)
    epochs = epoch_curve(parts)
    sample_runs, sample_summary = sample_size_curve(parts)

    arch_runs.to_csv(TABLES_DIR / "design_capacity_architecture_runs.csv", index=False)
    arch_summary.to_csv(TABLES_DIR / "design_capacity_architecture_summary.csv", index=False)
    epochs.to_csv(TABLES_DIR / "design_capacity_epoch_curve.csv", index=False)
    sample_summary.to_csv(TABLES_DIR / "design_capacity_sample_curve.csv", index=False)
    figure(arch_summary, epochs, sample_summary)

    best = arch_summary.iloc[0]
    canonical_row = arch_summary[arch_summary["arquitectura"] == "64-32"].iloc[0]
    result = {
        "schema_version": 1,
        "generated_by": "src/capacity_design_audit.py",
        "architectures_evaluated": len(ARCHITECTURE_GRID),
        "seeds": list(SEEDS),
        "selection_surface": "fit 2021 / select 2022 (same as the canonical selection)",
        "best_architecture_on_selection": str(best["arquitectura"]),
        "best_median_pr_auc": float(best["pr_auc_mediana"]),
        "canonical_median_pr_auc": float(canonical_row["pr_auc_mediana"]),
        "canonical_is_best": bool(best["arquitectura"] == "64-32"),
        "margin_over_canonical": float(best["pr_auc_mediana"] - canonical_row["pr_auc_mediana"]),
        "canonical_iqr": float(canonical_row["pr_auc_iqr"]),
        "epoch_curve_is_diagnostic_not_selection": True,
        "reference_period_opened": False,
        "canonical_artifacts_modified": False,
    }
    (TABLES_DIR / "design_capacity_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
