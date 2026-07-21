"""Validation-only audit of the definitive neural-network design.

This script supplies reproducible evidence for three questions that the frozen
bundle alone cannot answer: whether its regularizers help, whether several
networks are justified, and whether the network adds information beyond the
single strongest scene count.  All design comparisons use 2021--2022 for
fitting and 2023 for evaluation.  The historical 2024--2025 labels are opened
only after every rule and threshold has been fixed, and are used solely for
descriptive annual stability and the predeclared ``n_personas`` reference.

The canonical model, scaler, encoders and calibrator are read-only inputs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_e_modeling import choose_threshold, class_weights, evaluate, set_global_seed
from src.model_protocol import COMPANION_COUNT_COLUMNS, EXCLUDED_COLUMNS, transform_features


BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
FINAL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
SEEDS = (42, 314, 2718)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260721


@dataclass(frozen=True)
class AuditConfig:
    audit_id: str
    dropout: float
    l2: float


@dataclass(frozen=True)
class FrozenDesignDecisions:
    """Immutable boundary that must exist before the historical endpoint opens."""

    n_personas_threshold: float
    network_strategy: str
    ensemble_all_reported_ci_include_zero: bool


REGULARIZATION_CONFIGS = (
    AuditConfig("L2 + dropout (canónica)", 0.25, 1e-4),
    AuditConfig("Solo L2", 0.0, 1e-4),
    AuditConfig("Solo dropout", 0.25, 0.0),
    AuditConfig("Sin L2 ni dropout", 0.0, 0.0),
)


def _read_design_period(path: Path = BASE_PATH) -> pd.DataFrame:
    """Read only 2021--2023; the endpoint is excluded at the Parquet boundary."""
    frame = pd.read_parquet(path, filters=[("FECHA", "<", datetime(2024, 1, 1))])
    dates = pd.to_datetime(frame["FECHA"], errors="coerce")
    if dates.isna().any() or frame.empty or int(dates.dt.year.max()) > 2023:
        raise RuntimeError("Design-period read crossed the frozen 2023 boundary.")
    return frame


def _split_design_period(frame: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Create only train and validation partitions; no endpoint object exists here."""
    frame = frame.copy()
    frame["FECHA"] = pd.to_datetime(frame["FECHA"], errors="coerce")
    if frame["FECHA"].isna().any() or "target_multifatal" not in frame:
        raise ValueError("Design audit requires valid FECHA and target_multifatal fields.")
    years = frame["FECHA"].dt.year
    train = frame[years.isin([2021, 2022])].copy()
    validation = frame[years.eq(2023)].copy()
    if train.empty or validation.empty or not years.isin([2021, 2022, 2023]).all():
        raise ValueError("Design audit requires non-empty 2021--22 and 2023 partitions only.")

    def xy(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = partition.drop(columns=[column for column in EXCLUDED_COLUMNS if column in partition])
        labels = partition["target_multifatal"].astype("int8")
        return features, labels

    X_train, y_train = xy(train)
    X_validation, y_validation = xy(validation)
    return {
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_validation_raw": X_validation,
        "y_validation": y_validation,
    }


def _freeze_design_decisions(
    X_validation_raw: pd.DataFrame,
    y_validation: pd.Series,
    strategy_bootstrap: pd.DataFrame,
) -> FrozenDesignDecisions:
    """Freeze every rule that could otherwise adapt after endpoint inspection."""
    validation_score = pd.to_numeric(X_validation_raw["n_personas"], errors="coerce").fillna(0).to_numpy(float)
    person_rule = _choose_person_threshold(np.asarray(y_validation, dtype=int), validation_score)
    all_ci_include_zero = bool(
        ((strategy_bootstrap["ci_2_5"] <= 0) & (strategy_bootstrap["ci_97_5"] >= 0)).all()
    )
    strategy = (
        "retain_single_seed314_frozen"
        if all_ci_include_zero
        else "new_holdout_required_before_strategy_change"
    )
    return FrozenDesignDecisions(
        n_personas_threshold=float(person_rule["threshold_n_personas"]),
        network_strategy=strategy,
        ensemble_all_reported_ci_include_zero=all_ci_include_zero,
    )


def _read_reference_after_freeze(
    frozen: FrozenDesignDecisions,
    path: Path = BASE_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Open 2024--2025 only when an immutable validation decision exists."""
    if not isinstance(frozen, FrozenDesignDecisions):
        raise RuntimeError("Historical reference access requires frozen validation-only decisions.")
    frame = pd.read_parquet(path, filters=[("FECHA", ">=", datetime(2024, 1, 1))])
    dates = pd.to_datetime(frame["FECHA"], errors="coerce")
    if dates.isna().any() or frame.empty or not dates.dt.year.isin([2024, 2025]).all():
        raise RuntimeError("Historical reference read crossed the 2024--2025 boundary.")
    features = frame.drop(columns=[column for column in EXCLUDED_COLUMNS if column in frame])
    labels = frame["target_multifatal"].astype("int8")
    probabilities = pd.read_csv(TABLES_DIR / "final_reference_probabilities_2024_2025.csv")
    if len(probabilities) != len(features):
        raise RuntimeError("Historical reference rows do not align with frozen probabilities.")
    return features, labels, probabilities


def _callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=8, min_lr=1e-5
        ),
    ]


def _build_standard(feature_count: int, config: AuditConfig) -> keras.Model:
    model = keras.Sequential([layers.Input((feature_count,))])
    for units in (32, 16):
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(config.l2) if config.l2 else None,
            )
        )
        if config.dropout:
            model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _build_multibranch(feature_count: int, companion_indices: list[int]) -> keras.Model:
    context_indices = [index for index in range(feature_count) if index not in companion_indices]
    inputs = keras.Input((feature_count,), name="processed_features")
    context = layers.Lambda(lambda values: tf.gather(values, context_indices, axis=1), name="context_162")(inputs)
    companion = layers.Lambda(lambda values: tf.gather(values, companion_indices, axis=1), name="companion_13")(inputs)
    context = layers.Dense(24, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(context)
    companion = layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(companion)
    joined = layers.Concatenate()([context, companion])
    joined = layers.Dropout(0.25)(joined)
    joined = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(joined)
    joined = layers.Dropout(0.25)(joined)
    outputs = layers.Dense(1, activation="sigmoid")(joined)
    model = keras.Model(inputs, outputs, name="MLP_multirrama_162_13")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _fit(model: keras.Model, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> int:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=180,
        batch_size=64,
        class_weight=class_weights(y_train),
        callbacks=_callbacks(),
        verbose=0,
    )
    return len(history.history["loss"])


def _run_regularization(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    canonical_predictions: dict[int, np.ndarray] = {}
    for config in REGULARIZATION_CONFIGS:
        for seed in SEEDS:
            keras.backend.clear_session()
            set_global_seed(seed)
            model = _build_standard(X_train.shape[1], config)
            epochs = _fit(model, X_train, y_train, X_val, y_val)
            probabilities = model.predict(X_val, verbose=0).reshape(-1)
            threshold = choose_threshold(y_val, probabilities)
            rows.append(
                {
                    **asdict(config),
                    "seed": seed,
                    "epochs_ran": epochs,
                    "threshold_validation": float(threshold["threshold"]),
                    **evaluate(y_val, probabilities, float(threshold["threshold"])),
                }
            )
            if config == REGULARIZATION_CONFIGS[0]:
                canonical_predictions[seed] = probabilities
    runs = pd.DataFrame(rows)
    metrics = ["pr_auc", "roc_auc", "f1_multifatal", "precision_multifatal", "recall_multifatal"]
    summary_rows: list[dict[str, Any]] = []
    for audit_id, group in runs.groupby("audit_id", sort=False):
        row: dict[str, Any] = {
            "audit_id": audit_id,
            "dropout": float(group["dropout"].iloc[0]),
            "l2": float(group["l2"].iloc[0]),
            "seeds": len(group),
        }
        for metric in metrics:
            row[f"median_{metric}"] = float(group[metric].median())
            row[f"iqr_{metric}"] = float(group[metric].quantile(0.75) - group[metric].quantile(0.25))
        summary_rows.append(row)
    return runs, pd.DataFrame(summary_rows), canonical_predictions


def _metric_vector(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "f1_multifatal": float(f1_score(y, prediction, zero_division=0)),
    }


def _paired_bootstrap(
    y: np.ndarray,
    single: np.ndarray,
    ensemble: np.ndarray,
    single_threshold: float,
    ensemble_threshold: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {metric: [] for metric in ("pr_auc", "roc_auc", "f1_multifatal")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        left = _metric_vector(y[indices], single[indices], single_threshold)
        right = _metric_vector(y[indices], ensemble[indices], ensemble_threshold)
        for metric in samples:
            samples[metric].append(right[metric] - left[metric])
    estimates_left = _metric_vector(y, single, single_threshold)
    estimates_right = _metric_vector(y, ensemble, ensemble_threshold)
    return pd.DataFrame(
        [
            {
                "comparison": "ensemble_3_seeds - single_seed314",
                "partition": "validation_2023",
                "metric": metric,
                "single_estimate": estimates_left[metric],
                "ensemble_estimate": estimates_right[metric],
                "delta": estimates_right[metric] - estimates_left[metric],
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED,
            }
            for metric, values in samples.items()
        ]
    )


def _network_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    seed_predictions: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use the actual frozen seed-314 model for the deployed single-network row.
    frozen = keras.models.load_model(FINAL_DIR / "model.keras")
    single = frozen.predict(X_val, verbose=0).reshape(-1)
    seed_predictions = dict(seed_predictions)
    seed_predictions[314] = single
    ensemble = np.mean(np.vstack([seed_predictions[seed] for seed in SEEDS]), axis=0)
    single_threshold = float(choose_threshold(y_val, single)["threshold"])
    ensemble_threshold = float(choose_threshold(y_val, ensemble)["threshold"])

    companion_names = [*COMPANION_COUNT_COLUMNS, "edad_media_involucrados", "edad_faltante"]
    companion_indices = [feature_names.index(name) for name in companion_names]
    if len(companion_indices) != 13 or X_train.shape[1] - len(companion_indices) != 162:
        raise RuntimeError("The multibranch audit requires the frozen 162+13 feature partition.")

    rows = [
        {
            "strategy": "single_seed314_frozen",
            "seeds": "314",
            "threshold_validation": single_threshold,
            **evaluate(y_val, single, single_threshold),
        },
        {
            "strategy": "ensemble_mean_3_seeds",
            "seeds": "42,314,2718",
            "threshold_validation": ensemble_threshold,
            **evaluate(y_val, ensemble, ensemble_threshold),
        },
    ]
    branch_probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        keras.backend.clear_session()
        set_global_seed(seed)
        model = _build_multibranch(X_train.shape[1], companion_indices)
        _fit(model, X_train, y_train, X_val, y_val)
        branch_probabilities.append(model.predict(X_val, verbose=0).reshape(-1))
    branch = np.mean(np.vstack(branch_probabilities), axis=0)
    branch_threshold = float(choose_threshold(y_val, branch)["threshold"])
    rows.append(
        {
            "strategy": "multibranch_162_context_13_companion_mean_3_seeds",
            "seeds": "42,314,2718",
            "threshold_validation": branch_threshold,
            **evaluate(y_val, branch, branch_threshold),
        }
    )
    bootstrap = _paired_bootstrap(
        np.asarray(y_val, dtype=int), single, ensemble, single_threshold, ensemble_threshold
    )
    return pd.DataFrame(rows), bootstrap


def _choose_person_threshold(y: np.ndarray, n_personas: np.ndarray) -> dict[str, float]:
    candidates = sorted({int(value) for value in n_personas if np.isfinite(value)})
    rows = []
    for threshold in candidates:
        prediction = n_personas >= threshold
        rows.append(
            {
                "threshold_n_personas": threshold,
                "f1_multifatal": float(f1_score(y, prediction, zero_division=0)),
                "precision_multifatal": float(precision_score(y, prediction, zero_division=0)),
                "recall_multifatal": float(recall_score(y, prediction, zero_division=0)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["f1_multifatal"], row["recall_multifatal"], row["precision_multifatal"], row["threshold_n_personas"]
        ),
        reverse=True,
    )[0]


def _person_baseline(
    X_reference_raw: pd.DataFrame,
    y_reference: pd.Series,
    canonical_reference: pd.DataFrame,
    frozen: FrozenDesignDecisions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_score = pd.to_numeric(X_reference_raw["n_personas"], errors="coerce").fillna(0).to_numpy(float)
    y_ref = np.asarray(y_reference, dtype=int)
    threshold = frozen.n_personas_threshold
    baseline = {
        "model": "regla_n_personas",
        "probability_scale": "ordinal_count_not_probability",
        "selection_partition": "validation_2023_only",
        "threshold": threshold,
        **evaluate(y_ref, reference_score, threshold),
    }
    mlp_probability = canonical_reference["raw_probability"].to_numpy(float)
    mlp_threshold = float(canonical_reference["raw_threshold"].iloc[0])
    mlp = {
        "model": "MLP_canónica_cruda",
        "probability_scale": "raw_mlp_sigmoid",
        "selection_partition": "validation_2023_only",
        "threshold": mlp_threshold,
        **evaluate(y_ref, mlp_probability, mlp_threshold),
    }
    comparison = pd.DataFrame([baseline, mlp])

    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    values = {metric: [] for metric in ("pr_auc", "roc_auc", "f1_multifatal")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(y_ref), len(y_ref))
        if np.unique(y_ref[indices]).size < 2:
            continue
        mlp_metrics = _metric_vector(y_ref[indices], mlp_probability[indices], mlp_threshold)
        baseline_metrics = _metric_vector(y_ref[indices], reference_score[indices], threshold)
        for metric in values:
            values[metric].append(mlp_metrics[metric] - baseline_metrics[metric])
    paired = pd.DataFrame(
        [
            {
                "comparison": "MLP_canónica_cruda - regla_n_personas",
                "partition": "historical_reference_2024_2025_post_hoc",
                "metric": metric,
                "delta": float(mlp[metric] - baseline[metric]),
                "ci_2_5": float(np.quantile(samples, 0.025)),
                "ci_97_5": float(np.quantile(samples, 0.975)),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "threshold_n_personas_selected_on_2023": threshold,
            }
            for metric, samples in values.items()
        ]
    )
    return comparison, paired


def _annual_stability(probabilities: pd.DataFrame) -> pd.DataFrame:
    probabilities = probabilities.copy()
    probabilities["year"] = pd.to_datetime(probabilities["fecha"]).dt.year
    rows = []
    for year, group in probabilities.groupby("year"):
        y = group["actual_multifatal"].to_numpy(int)
        score = group["calibrated_probability"].to_numpy(float)
        threshold = float(group["calibrated_threshold"].iloc[0])
        prediction = score >= threshold
        rows.append(
            {
                "year": int(year),
                "status": "historical_reference_descriptive_only",
                "n": len(group),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "threshold": threshold,
                "f1_multifatal": float(f1_score(y, prediction, zero_division=0)),
                "precision_multifatal": float(precision_score(y, prediction, zero_division=0)),
                "recall_multifatal": float(recall_score(y, prediction, zero_division=0)),
                "pr_auc": float(average_precision_score(y, score)),
                "roc_auc": float(roc_auc_score(y, score)),
            }
        )
    return pd.DataFrame(rows)


def _figure(reg_summary: pd.DataFrame, strategies: pd.DataFrame, paired: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#C94F16", "#496B7C", "#7E9187", "#B6AAA0"]
    axes[0].barh(reg_summary["audit_id"], reg_summary["median_pr_auc"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Ablación de regularización · validación 2023")
    axes[0].set_xlabel("Mediana PR-AUC · 3 semillas")
    axes[0].axvline(0.098, color="#202522", linestyle="--", linewidth=1, label="prevalencia")
    axes[0].legend(frameon=False, fontsize=8)

    labels = ["1 red\ncanónica", "Ensemble\n3 semillas", "Multirrama\n162 + 13"]
    axes[1].bar(labels, strategies["pr_auc"], color=["#C94F16", "#496B7C", "#7E9187"])
    axes[1].set_ylim(0, max(.55, float(strategies["pr_auc"].max()) * 1.2))
    axes[1].set_title("Una vs. varias redes · validación 2023")
    axes[1].set_ylabel("PR-AUC")

    ordered = paired.set_index("metric").loc[["pr_auc", "roc_auc", "f1_multifatal"]].reset_index()
    y = np.arange(len(ordered))
    axes[2].errorbar(
        ordered["delta"], y,
        xerr=[ordered["delta"] - ordered["ci_2_5"], ordered["ci_97_5"] - ordered["delta"]],
        fmt="D", color="#C94F16", ecolor="#496B7C", capsize=4,
    )
    axes[2].axvline(0, color="#202522", linewidth=1)
    axes[2].set_yticks(y, ["PR-AUC", "ROC-AUC", "F1"])
    axes[2].set_title("Ensemble − 1 red · bootstrap pareado 95 %")
    axes[2].set_xlabel("Diferencia en validación 2023")
    fig.suptitle("Auditoría reproducible del diseño neuronal", fontsize=16, fontweight="bold")
    fig.text(
        .5,
        .015,
        "La selección numérica y la recalibración del modelo definitivo usan 2021–2023; 2024–2025 es una segunda consulta de referencia.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(FIGURES_DIR / "design_validation_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, Any]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    design_frame = _read_design_period(BASE_PATH)
    splits = _split_design_period(design_frame)
    scaler = joblib.load(FINAL_DIR / "scaler.joblib")
    encoders = joblib.load(FINAL_DIR / "encoders.joblib")
    feature_names = json.loads((FINAL_DIR / "feature_list.json").read_text(encoding="utf-8"))
    X_train = transform_features(splits["X_train_raw"], scaler, encoders)  # type: ignore[arg-type]
    X_val = transform_features(splits["X_validation_raw"], scaler, encoders)  # type: ignore[arg-type]
    y_train = splits["y_train"]
    y_val = splits["y_validation"]

    regularization_runs, regularization_summary, seed_predictions = _run_regularization(
        X_train, y_train, X_val, y_val  # type: ignore[arg-type]
    )
    strategies, strategy_bootstrap = _network_strategies(
        X_train, y_train, X_val, y_val, feature_names, seed_predictions  # type: ignore[arg-type]
    )

    # This immutable object is the endpoint gate. Only after it exists can the
    # second Parquet read materialize 2024--2025 features or labels.
    frozen = _freeze_design_decisions(
        splits["X_validation_raw"], y_val, strategy_bootstrap  # type: ignore[arg-type]
    )
    X_reference_raw, y_reference, reference_probabilities = _read_reference_after_freeze(frozen, BASE_PATH)
    person_comparison, person_bootstrap = _person_baseline(
        X_reference_raw, y_reference, reference_probabilities, frozen
    )
    annual = _annual_stability(reference_probabilities)

    outputs = {
        "design_regularization_runs.csv": regularization_runs,
        "design_regularization_summary.csv": regularization_summary,
        "design_network_strategy_validation.csv": strategies,
        "design_network_strategy_bootstrap.csv": strategy_bootstrap,
        "design_n_personas_reference_comparison.csv": person_comparison,
        "design_n_personas_paired_bootstrap.csv": person_bootstrap,
        "design_annual_stability_2024_2025.csv": annual,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLES_DIR / name, index=False)
    _figure(regularization_summary, strategies, strategy_bootstrap)

    result = {
        "schema_version": 1,
        "generated_by": "src/validation_design_audit.py",
        "selection_boundary": {
            "fit": "2021-2022",
            "design_evaluation": "2023",
            "historical_reference": "2024-2025, opened only after design rules were frozen",
        },
        "raw_input_fields": 26,
        "processed_features": len(feature_names),
        "regularization_variants": len(REGULARIZATION_CONFIGS),
        "regularization_seeds": list(SEEDS),
        "network_strategy_conclusion": (
            "retain the single frozen seed-314 MLP; the ensemble is a future hypothesis requiring "
            "a new holdout and independent calibration"
        ),
        "ensemble_all_reported_ci_include_zero": frozen.ensemble_all_reported_ci_include_zero,
        "n_personas_rule_selected_on_validation": int(frozen.n_personas_threshold),
        "reference_is_post_hoc": True,
        "canonical_artifacts_modified": False,
    }
    (TABLES_DIR / "design_validation_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
