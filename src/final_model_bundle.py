"""Build and load the definitive Accident_NN model bundle.

This module never trains or selects an MLP. It completes the already-selected
network with validation-only calibration, generates the historical reference
evaluation, and records hashes for every runtime artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/accident_nn_matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from src.block_e_modeling import MLP_GRID, _platt_features, choose_threshold, ece
from src.chronology import (
    CALIBRATED_THRESHOLD_VALIDATION_PARTITION,
    CALIBRATION_THRESHOLD_VALIDATION_PARTITION,
    RAW_THRESHOLD_VALIDATION_PARTITION,
    protocol_chronology_markdown,
)
from src.model_protocol import (
    EXCLUDED_COLUMNS,
    transform_features,
)
from src.source_provenance import source_manifest_hash, verify_raw_sources


BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
FINAL_MODEL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"
MODEL_VERSION = "canonical-3.0.0"
CALIBRATION_FOLDS = 5
CALIBRATION_SEED = 20260709
BOOTSTRAP_ITERATIONS = 1000
ENDPOINT_START = pd.Timestamp("2024-01-01")
ENDPOINT_END = pd.Timestamp("2026-01-01")
CALIBRATION_POLICY = (
    "Choose the lowest 5-fold stratified OOF Brier score on 2023 validation; "
    "ties within 1e-12 use lower ECE, then prefer Platt for lower variance. "
    "Platt uses L2-regularized logistic regression with C=1.0 and liblinear."
)
REFERENCE_MODEL_NAMES = {
    "MLP_definitiva": "MLP cruda",
    "LogisticRegression_balanced": "regresión logística",
    "RandomForest_balanced": "Random Forest",
}
DESIGN_EVIDENCE_PATHS = (
    Path("report/tables/design_annual_stability_2024_2025.csv"),
    Path("report/tables/design_network_strategy_bootstrap.csv"),
    Path("report/tables/design_network_strategy_validation.csv"),
    Path("report/tables/design_regularization_runs.csv"),
    Path("report/tables/design_regularization_summary.csv"),
    Path("report/tables/design_validation_audit.json"),
    Path("report/tables/personas_target_proxy_identity.csv"),
    Path("report/tables/personas_target_proxy_identity.json"),
    Path("report/figures/design_validation_evidence.png"),
)


def _xy_partition(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return (
        partition.drop(columns=[column for column in EXCLUDED_COLUMNS if column in partition]),
        partition["target_multifatal"].astype("int8"),
    )


def _read_design_before_endpoint(
    base_path: Path,
    reader: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series]]:
    """Read only 2021--2023; endpoint rows must not be materialized here."""
    design = reader(base_path, filters=[("FECHA", "<", ENDPOINT_START)])
    design = design.copy()
    design["FECHA"] = pd.to_datetime(design["FECHA"], errors="coerce")
    if (
        design.empty
        or design["FECHA"].isna().any()
        or design["FECHA"].ge(ENDPOINT_START).any()
        or not design["FECHA"].is_monotonic_increasing
    ):
        raise RuntimeError("The pre-endpoint read materialized invalid or 2024+ rows.")
    years = design["FECHA"].dt.year
    train = design[years.isin([2021, 2022])].copy()
    validation = design[years.eq(2023)].copy()
    if min(len(train), len(validation)) == 0:
        raise RuntimeError("The design protocol requires non-empty 2021--22 and 2023 partitions.")
    X_train, y_train = _xy_partition(train)
    X_validation, y_validation = _xy_partition(validation)
    return design, {
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_validation_raw": X_validation,
        "y_validation": y_validation,
    }


def _read_endpoint_after_freeze(
    base_path: Path,
    final_dir: Path,
    frozen_decision_hashes: dict[str, str],
    reader: Callable[..., pd.DataFrame],
    endpoint_opened_hook: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Open 2024--2025 only after every decision artifact is present and frozen."""
    required = {
        "calibration_selection.json",
        "calibrator.joblib",
        "thresholds.json",
        "feature_schema.json",
    }
    if set(frozen_decision_hashes) != required:
        raise RuntimeError("Endpoint gate requires the complete frozen decision artifact set.")
    for name, expected in frozen_decision_hashes.items():
        path = final_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Endpoint gate rejected an unfrozen decision artifact: {name}")
    if endpoint_opened_hook is not None:
        endpoint_opened_hook()
    endpoint = reader(
        base_path,
        filters=[("FECHA", ">=", ENDPOINT_START), ("FECHA", "<", ENDPOINT_END)],
    ).copy()
    endpoint["FECHA"] = pd.to_datetime(endpoint["FECHA"], errors="coerce")
    if (
        endpoint.empty
        or endpoint["FECHA"].isna().any()
        or endpoint["FECHA"].lt(ENDPOINT_START).any()
        or endpoint["FECHA"].ge(ENDPOINT_END).any()
        or not endpoint["FECHA"].is_monotonic_increasing
    ):
        raise RuntimeError("The endpoint read must contain only 2024--2025 rows.")
    return _xy_partition(endpoint)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_evidence_hashes(root: Path = ROOT, *, require_complete: bool = True) -> dict[str, str]:
    """Hash validation-audit evidence in one stable, reproducible order."""
    missing = [relative for relative in DESIGN_EVIDENCE_PATHS if not (root / relative).is_file()]
    if missing and require_complete:
        raise RuntimeError(
            "Generate design evidence before the canonical manifest: "
            + ", ".join(str(path) for path in missing)
        )
    hashes: dict[str, str] = {}
    for relative in DESIGN_EVIDENCE_PATHS:
        path = root / relative
        if not path.is_file():
            continue
        key = str(relative) if relative.parts[1] == "figures" else relative.name
        hashes[key] = sha256_file(path)
    return dict(sorted(hashes.items()))


def reference_metric_leaders(comparison: pd.DataFrame) -> dict[str, dict[str, str | float]]:
    """Return nominal metric leaders from the comparable raw-score rows."""
    raw = comparison[comparison["probability_scale"] == "raw"].copy()
    raw = raw[raw["model"].isin(REFERENCE_MODEL_NAMES)].copy()
    missing = set(REFERENCE_MODEL_NAMES) - set(raw["model"])
    if missing:
        raise ValueError(f"Reference comparison is missing canonical models: {sorted(missing)}")
    leaders: dict[str, dict[str, str | float]] = {}
    for metric in ("pr_auc", "roc_auc", "f1_multifatal"):
        row = raw.loc[raw[metric].idxmax()]
        model = str(row["model"])
        leaders[metric] = {
            "model": model,
            "display_name": REFERENCE_MODEL_NAMES[model],
            "value": float(row[metric]),
        }
    return leaders


def canonical_protocol_text(
    *,
    method: str,
    raw_threshold: float,
    calibrated_threshold: float,
    comparison: pd.DataFrame,
    paired_summary: dict[str, Any],
) -> str:
    leaders = reference_metric_leaders(comparison)
    raw_mlp = comparison[
        (comparison["model"] == "MLP_definitiva")
        & (comparison["probability_scale"] == "raw")
    ].iloc[0]
    rf_f1 = paired_summary["results"]["RandomForest_balanced"]["f1_multifatal"]
    return f"""# Protocolo del modelo canónico {MODEL_VERSION}

## Alcance
La salida estima retrospectivamente multifatalidad (2 o más fallecidos) condicionada a un siniestro ya fatal registrado. Todo agregado de PERSONAS está excluido porque su cardinalidad y el conteo de fallecidos reconstruyen el objetivo. La extracción no aporta timestamps por variable para afirmar disponibilidad al instante de notificación y no demuestra causalidad.

## Cronología canónica
{protocol_chronology_markdown()}
- Épocas del reajuste final: 14, fijadas por el protocolo de selección de 2022.
- Calibración: comparación Platt/isotónica con 5 folds OOF estratificados y criterio de menor Brier OOF. Método resultante: {method}.
- Umbral crudo y calibrado: máximo F1 de validación; el calibrado usa las probabilidades OOF del método resultante. Ambos aplican los desempates predeclarados.

## Artefacto congelado
La arquitectura y los pesos no se reentrenan ni se vuelven a buscar al materializar `models/final/`. Los umbrales crudo ({raw_threshold:.2f}) y calibrado ({calibrated_threshold:.2f}) pertenecen a escalas distintas y nunca deben intercambiarse. La finalidad del artefacto de entrega no convierte 2024--2025 en una cohorte confirmatoria independiente.

## Lectura honesta
En la referencia 2024--2025 la MLP cruda obtiene PR-AUC {raw_mlp['pr_auc']:.4f}, ROC-AUC {raw_mlp['roc_auc']:.4f} y F1 {raw_mlp['f1_multifatal']:.4f}. Los liderazgos nominales derivados de la tabla canónica son: PR-AUC, {leaders['pr_auc']['display_name']} ({leaders['pr_auc']['value']:.4f}); ROC-AUC, {leaders['roc_auc']['display_name']} ({leaders['roc_auc']['value']:.4f}); F1, {leaders['f1_multifatal']['display_name']} ({leaders['f1_multifatal']['value']:.4f}). Frente a Random Forest, PR-AUC y ROC-AUC no muestran diferencia nominal; F1 sí favorece nominalmente a la MLP por {rf_f1['delta_mlp_minus_baseline']:.4f}, IC95% [{rf_f1['delta_ci95_low']:.4f}, {rf_f1['delta_ci95_high']:.4f}]. Son seis comparaciones con intervalos nominales al 95% sin ajuste por multiplicidad; no se ofrece cobertura familiar simultánea ni se prueba superioridad universal.
"""


def canonical_claims(comparison: pd.DataFrame, paired_summary: dict[str, Any]) -> dict[str, str]:
    leaders = reference_metric_leaders(comparison)
    rf_f1 = paired_summary["results"]["RandomForest_balanced"]["f1_multifatal"]
    rf_pr = paired_summary["results"]["RandomForest_balanced"]["pr_auc"]
    rf_roc = paired_summary["results"]["RandomForest_balanced"]["roc_auc"]
    if (
        paired_summary.get("comparison_family_size") != 6
        or paired_summary.get("multiplicity_adjustment") != "none"
        or paired_summary.get("simultaneous_familywise_coverage") is not False
    ):
        raise RuntimeError("Paired-comparison multiplicity metadata is incomplete.")
    if not rf_f1.get("nominal_significant_at_5pct") or rf_pr.get("nominal_significant_at_5pct") or rf_roc.get("nominal_significant_at_5pct"):
        raise RuntimeError("Paired Random Forest conclusions do not match the canonical metric-level evidence.")
    return {
        "supported": (
            "On this fixed historical reference, nominal leaders derived from the canonical raw-score comparison are "
            f"PR-AUC: {leaders['pr_auc']['display_name']} ({leaders['pr_auc']['value']:.4f}); "
            f"ROC-AUC: {leaders['roc_auc']['display_name']} ({leaders['roc_auc']['value']:.4f}); "
            f"F1: {leaders['f1_multifatal']['display_name']} ({leaders['f1_multifatal']['value']:.4f})."
        ),
        "not_supported": (
            "Universal superiority is not established: paired PR-AUC and ROC-AUC differences against Random Forest "
            "include zero. F1 is the nominal exception and favors the MLP by "
            f"{rf_f1['delta_mlp_minus_baseline']:.4f} (95% CI "
            f"[{rf_f1['delta_ci95_low']:.5f}, {rf_f1['delta_ci95_high']:.5f}]). "
            "All six paired intervals are nominal and unadjusted for multiplicity; no simultaneous familywise coverage is claimed."
        ),
    }


def _fit_calibrator(method: str, probabilities: np.ndarray, labels: np.ndarray) -> object:
    if method == "platt":
        return LogisticRegression(C=1.0, solver="liblinear", random_state=42).fit(
            _platt_features(probabilities), labels
        )
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator: object, method: str, probabilities: np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float).reshape(-1)
    if method == "platt":
        calibrated = calibrator.predict_proba(_platt_features(raw))[:, 1]  # type: ignore[attr-defined]
    elif method == "isotonic":
        calibrated = calibrator.predict(raw)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def select_calibrator_validation_only(
    raw_validation_probabilities: np.ndarray,
    validation_labels: pd.Series | np.ndarray,
    folds: int = CALIBRATION_FOLDS,
    seed: int = CALIBRATION_SEED,
) -> tuple[str, object, dict[str, Any], np.ndarray]:
    """Select calibration and calibrated threshold using only validation data."""
    probabilities = np.asarray(raw_validation_probabilities, dtype=float).reshape(-1)
    labels = np.asarray(validation_labels, dtype=int).reshape(-1)
    if len(probabilities) != len(labels):
        raise ValueError("Validation probabilities and labels must have equal length.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration selection requires both classes in validation.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    method_results: dict[str, dict[str, Any]] = {}
    method_oof: dict[str, np.ndarray] = {}
    for method in ("platt", "isotonic"):
        oof = np.full(len(labels), np.nan, dtype=float)
        for fit_indices, held_out_indices in splitter.split(probabilities, labels):
            calibrator = _fit_calibrator(method, probabilities[fit_indices], labels[fit_indices])
            oof[held_out_indices] = apply_calibrator(calibrator, method, probabilities[held_out_indices])
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Non-finite OOF predictions for {method}.")
        threshold = choose_threshold(pd.Series(labels), oof)
        method_oof[method] = oof
        method_results[method] = {
            "oof_brier": float(brier_score_loss(labels, oof)),
            "oof_ece_10_bins": ece(labels, oof),
            "oof_pr_auc": float(average_precision_score(labels, oof)),
            "oof_roc_auc": float(roc_auc_score(labels, oof)),
            "calibrated_threshold": threshold,
        }

    selected_method = sorted(
        method_results,
        key=lambda name: (
            round(float(method_results[name]["oof_brier"]), 12),
            round(float(method_results[name]["oof_ece_10_bins"]), 12),
            0 if name == "platt" else 1,
        ),
    )[0]
    final_calibrator = _fit_calibrator(selected_method, probabilities, labels)
    evidence = {
        "source_partition": CALIBRATION_THRESHOLD_VALIDATION_PARTITION,
        "reference_labels_used_for_calibration_or_threshold_validation": False,
        "folds": folds,
        "splitter": "StratifiedKFold(shuffle=True)",
        "seed": seed,
        "validation_policy": CALIBRATION_POLICY,
        "method_configuration": {
            "platt": "LogisticRegression(C=1.0, solver=liblinear, random_state=42) on clipped logits",
            "isotonic": "IsotonicRegression(out_of_bounds=clip)",
        },
        "selected_method": selected_method,
        "methods": method_results,
        "threshold_source": "resulting method OOF calibrated validation predictions",
        "final_fit": "resulting calibrator refit on all 2023 raw validation predictions",
    }
    return selected_method, final_calibrator, evidence, method_oof[selected_method]


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "class_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "f1_multifatal": float(f1_score(y_true, predictions, zero_division=0)),
        "precision_multifatal": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_multifatal": float(recall_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece_10_bins": ece(y_true, probabilities),
    }


def _bootstrap_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    scale: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    rng = np.random.default_rng(CALIBRATION_SEED)
    keys = ["f1_multifatal", "precision_multifatal", "recall_multifatal", "pr_auc", "roc_auc", "brier", "ece_10_bins"]
    values = {key: [] for key in keys}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        metrics = _metrics(sampled_y, probabilities[indices], threshold)
        for key in keys:
            values[key].append(float(metrics[key]))
    estimate = _metrics(y_true, probabilities, threshold)
    return pd.DataFrame(
        {
            "probability_scale": scale,
            "metric": keys,
            "estimate": [estimate[key] for key in keys],
            "ci_2_5": [float(np.quantile(values[key], 0.025)) for key in keys],
            "ci_97_5": [float(np.quantile(values[key], 0.975)) for key in keys],
            "bootstrap_iterations": iterations,
        }
    )


def _classification_tables(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float, scale: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    confusion = pd.DataFrame(
        [
            {"probability_scale": scale, "actual": actual, "predicted": predicted, "count": int(matrix[actual, predicted])}
            for actual in (0, 1)
            for predicted in (0, 1)
        ]
    )
    report = pd.DataFrame(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["single_fatality", "multifatal"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose().reset_index(names="class_or_average")
    report.insert(0, "probability_scale", scale)
    return confusion, report


def _input_schema(base: pd.DataFrame, feature_list: list[str]) -> dict[str, Any]:
    required = [
        "FECHA", "HORA", "DEPARTAMENTO", "CODIGO_VIA", "LATITUD", "LONGITUD",
        "CLASE", "ZONA", "RED_VIAL", "TIPO_VIA", "CLIMA", "CARACTERISTICA_VIA",
        "PERFIL_VIA", "SUPERFICIE",
        # Consolidated VEHICULOS aggregates only. PERSONAS is structurally banned.
        "n_vehiculos", "n_bus", "n_pesado_carga", "n_moto", "n_no_identificado",
        "n_interprovincial", "n_transporte_publico",
    ]
    non_nullable_runtime_fields = {"LATITUD", "LONGITUD", "n_vehiculos"}
    fields = []
    for column in required:
        fields.append(
            {
                "name": column,
                "training_dtype": str(base[column].dtype),
                "required": True,
                # Runtime inference requires a complete coordinate pair even
                # though the historical training source contains missingness.
                "nullable": False
                if column in non_nullable_runtime_fields
                else bool(base[column].isna().any()),
                "role": "runtime_input",
            }
        )
    return {
        "schema_version": "2.0",
        "target": {
            "name": "target_multifatal",
            "definition": "1 when FALLECIDOS >= 2, conditional on an already-fatal crash",
        },
        "required_raw_fields": fields,
        "runtime_constraints": {
            "coordinate_pair": {
                "fields": ["LATITUD", "LONGITUD"],
                "required": True,
                "nullable": False,
            }
        },
        "excluded_leakage_columns": sorted(EXCLUDED_COLUMNS),
        "processed_feature_count": len(feature_list),
        "processed_feature_dtype": "float32",
        "processed_feature_order": feature_list,
        "forbidden_source_families": {
            "PERSONAS": ["n_personas", "n_pasajeros", "n_peatones", "n_conductor_fugado", "edad_media_involucrados", "edad_faltante"]
        },
        "scope": "retrospective historical classification on consolidated records; field-level availability timestamps are absent",
    }


def _baseline_reference(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    # liblinear is numerically stable on the transformed design matrix when
    # using float64; float32 can overflow in BLAS gradient products on macOS.
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")
    validation = pd.read_csv(TABLES_DIR / "model_selection_baseline_validation.csv").set_index("model")
    models: dict[str, object] = {
        "DummyClassifier_prior": DummyClassifier(strategy="prior", random_state=42),
        "LogisticRegression_balanced": LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)  # type: ignore[attr-defined]
        # Apple Accelerate can emit spurious matmul overflow warnings from
        # sklearn's safe_sparse_dot even though coefficients and probabilities
        # are finite. Suppress only that narrow warning and validate outputs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
            probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]  # type: ignore[attr-defined]
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Baseline {name} produced non-finite probabilities.")
        threshold = float(validation.loc[name, "threshold"])
        rows.append(
            {
                "model": name,
                "probability_scale": "raw",
                "threshold_source": "2023 threshold-validation policy",
                **_metrics(y_test, probabilities, threshold),
            }
        )
    return pd.DataFrame(rows)


def build_canonical_bundle(
    root: Path = ROOT,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    endpoint_opened_hook: Callable[[], None] | None = None,
    parquet_reader: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Complete the selected model bundle and create reference outputs."""
    base_path = root / "data" / "processed" / "base_limpia.parquet"
    final_dir = root / "models" / "final"
    tables_dir = root / "report" / "tables"
    final_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_verification = verify_raw_sources(root=root)

    reader = parquet_reader or pd.read_parquet
    design_base, splits = _read_design_before_endpoint(base_path, reader)
    scaler = joblib.load(final_dir / "scaler.joblib")
    encoders = joblib.load(final_dir / "encoders.joblib")
    feature_list = json.loads((final_dir / "feature_list.json").read_text(encoding="utf-8"))
    selection = json.loads((final_dir / "model_selection.json").read_text(encoding="utf-8"))
    model = keras.models.load_model(final_dir / "model.keras")
    if feature_list != encoders.get("feature_list"):
        raise ValueError("feature_list.json and encoders.joblib disagree.")
    if int(model.input_shape[-1]) != len(feature_list):
        raise ValueError("Frozen MLP input width does not match the frozen feature contract.")

    X_train = transform_features(splits["X_train_raw"], scaler, encoders)  # type: ignore[arg-type]
    X_validation = transform_features(splits["X_validation_raw"], scaler, encoders)  # type: ignore[arg-type]
    y_train = splits["y_train"]
    y_val = splits["y_validation"]
    raw_validation = model.predict(X_validation, verbose=0).reshape(-1)
    method, calibrator, calibration_evidence, oof_calibrated_validation = select_calibrator_validation_only(
        raw_validation, y_val
    )
    raw_threshold = float(selection["selected_threshold"]["threshold"])
    calibrated_selection = choose_threshold(y_val, oof_calibrated_validation)  # type: ignore[arg-type]
    calibrated_threshold = float(calibrated_selection["threshold"])
    calibration_evidence["selected_calibrated_threshold"] = calibrated_selection

    # Persist every calibration/threshold/schema decision before the endpoint
    # reader can materialize a single 2024--2025 row.
    (final_dir / "calibration_selection.json").write_text(
        json.dumps(calibration_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(calibrator, final_dir / "calibrator.joblib")
    thresholds = {
        "raw": {
            "value": raw_threshold,
            "probability_scale": "raw_mlp_sigmoid",
            "source_partition": RAW_THRESHOLD_VALIDATION_PARTITION,
            "policy": "maximum 2023 threshold-validation F1; ties recall, precision, higher threshold",
        },
        "calibrated": {
            "value": calibrated_threshold,
            "probability_scale": f"{method}_calibrated_probability",
            "source_partition": CALIBRATED_THRESHOLD_VALIDATION_PARTITION,
            "policy": str(calibrated_selection["selection_policy"]),
        },
    }
    (final_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = _input_schema(design_base, feature_list)
    (final_dir / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen_decision_hashes = {
        name: sha256_file(final_dir / name)
        for name in (
            "calibration_selection.json",
            "calibrator.joblib",
            "thresholds.json",
            "feature_schema.json",
        )
    }

    X_test_raw, endpoint_labels_series = _read_endpoint_after_freeze(
        base_path,
        final_dir,
        frozen_decision_hashes,
        reader,
        endpoint_opened_hook,
    )
    X_test = transform_features(X_test_raw, scaler, encoders)
    endpoint_labels = np.asarray(endpoint_labels_series, dtype=int)
    raw_test = model.predict(X_test, verbose=0).reshape(-1)
    calibrated_test = apply_calibrator(calibrator, method, raw_test)

    raw_metrics = _metrics(endpoint_labels, raw_test, raw_threshold)
    calibrated_metrics = _metrics(endpoint_labels, calibrated_test, calibrated_threshold)
    metrics = {
        "reference_partition": "2024-2025",
        "endpoint_status": "historical reference; labels have already been observed and cannot support further tuning",
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    (tables_dir / "final_reference_metrics_2024_2025.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    probability_table = pd.DataFrame(
        {
            # A filtered read of a parquet RangeIndex is re-based at zero by
            # pyarrow. Reconstruct the stable canonical row identifier from
            # the chronologically sorted design-prefix length without opening
            # the endpoint early.
            "row_index": np.arange(len(design_base), len(design_base) + len(X_test_raw), dtype=int),
            "fecha": pd.to_datetime(X_test_raw["FECHA"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "departamento": X_test_raw["DEPARTAMENTO"].to_numpy(),
            "actual_multifatal": endpoint_labels,
            "raw_probability": raw_test,
            "raw_threshold": raw_threshold,
            "raw_prediction": (raw_test >= raw_threshold).astype(int),
            "calibration_method": method,
            "calibrated_probability": calibrated_test,
            "calibrated_threshold": calibrated_threshold,
            "calibrated_prediction": (calibrated_test >= calibrated_threshold).astype(int),
        }
    )
    probability_table.to_csv(tables_dir / "final_reference_probabilities_2024_2025.csv", index=False)
    error_table = probability_table[
        probability_table["actual_multifatal"] != probability_table["calibrated_prediction"]
    ].copy()
    error_table["error_type"] = np.where(error_table["actual_multifatal"] == 1, "false_negative", "false_positive")
    error_table["distance_to_threshold"] = (
        error_table["calibrated_probability"] - calibrated_threshold
    ).abs()
    error_table = (
        error_table.sort_values(["error_type", "distance_to_threshold"])
        .groupby("error_type", group_keys=False)
        .head(50)
    )
    error_table.to_csv(
        tables_dir / "final_reference_error_examples_2024_2025.csv", index=False
    )

    confusion_parts, report_parts = [], []
    for scale, probabilities, threshold in (
        ("raw", raw_test, raw_threshold),
        ("calibrated", calibrated_test, calibrated_threshold),
    ):
        confusion, report = _classification_tables(endpoint_labels, probabilities, threshold, scale)
        confusion_parts.append(confusion)
        report_parts.append(report)
    pd.concat(confusion_parts, ignore_index=True).to_csv(
        tables_dir / "final_reference_confusion_matrix_2024_2025.csv", index=False
    )
    pd.concat(report_parts, ignore_index=True).to_csv(
        tables_dir / "final_reference_classification_report_2024_2025.csv", index=False
    )
    ci = pd.concat(
        [
            _bootstrap_metrics(endpoint_labels, raw_test, raw_threshold, "raw", bootstrap_iterations),
            _bootstrap_metrics(endpoint_labels, calibrated_test, calibrated_threshold, "calibrated", bootstrap_iterations),
        ],
        ignore_index=True,
    )
    ci.to_csv(tables_dir / "final_reference_bootstrap_ci_2024_2025.csv", index=False)
    bootstrap_metadata = {
        "schema_version": 1,
        "method": "row-level percentile bootstrap conditional on frozen predictions",
        "pipeline_refit_per_resample": False,
        "included_uncertainty": "row sampling variation within the already-observed historical reference",
        "excluded_uncertainty": [
            "training and model-selection uncertainty",
            "calibration and threshold-selection uncertainty",
            "temporal dependence",
            "spatial dependence",
            "repeated consultation of the reference period",
            "future or external generalization",
            "individual-prediction uncertainty",
        ],
    }
    (tables_dir / "final_reference_bootstrap_metadata.json").write_text(
        json.dumps(bootstrap_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = _baseline_reference(X_train, y_train, X_test, endpoint_labels)  # type: ignore[arg-type]
    mlp_rows = pd.DataFrame(
        [
            {"model": "MLP_definitiva", "probability_scale": "raw", "threshold_source": "2023 threshold-validation policy", **raw_metrics},
            {"model": "MLP_definitiva", "probability_scale": method, "threshold_source": "2023 OOF calibration/threshold-validation policy", **calibrated_metrics},
        ]
    )
    reference_comparison = pd.concat([mlp_rows, baseline], ignore_index=True)
    reference_comparison.to_csv(
        tables_dir / "final_reference_baseline_comparison_2024_2025.csv", index=False
    )
    paired_summary = json.loads(
        (tables_dir / "final_paired_bootstrap_2024_2025.json").read_text(encoding="utf-8")
    )
    protocol_text = canonical_protocol_text(
        method=method,
        raw_threshold=raw_threshold,
        calibrated_threshold=calibrated_threshold,
        comparison=reference_comparison,
        paired_summary=paired_summary,
    )
    (tables_dir / "final_model_protocol.md").write_text(protocol_text, encoding="utf-8")

    split_metadata = {}
    for name, labels, raw_partition in (
        ("train", splits["y_train"], splits["X_train_raw"]),
        ("validation", splits["y_validation"], splits["X_validation_raw"]),
        ("reference", endpoint_labels_series, X_test_raw),
    ):
        dates = pd.to_datetime(raw_partition["FECHA"])  # type: ignore[index]
        split_metadata[name] = {
            "date_min": dates.min().strftime("%Y-%m-%d"),
            "date_max": dates.max().strftime("%Y-%m-%d"),
            "count": int(len(labels)),  # type: ignore[arg-type]
            "positive_count": int(np.asarray(labels).sum()),
            "class_rate": float(np.asarray(labels).mean()),
        }
    selected_config_id = str(selection["selected_config"]["config_id"])
    if not any(config.config_id == selected_config_id for config in MLP_GRID):
        raise ValueError(f"Frozen selection references unknown config {selected_config_id}.")
    bundle_files = [
        "model.keras", "scaler.joblib", "encoders.joblib", "feature_list.json",
        "calibrator.joblib", "thresholds.json", "feature_schema.json", "calibration_selection.json",
    ]
    artifact_hashes = {name: sha256_file(final_dir / name) for name in bundle_files}
    reference_files = [
        "final_reference_metrics_2024_2025.json",
        "final_reference_probabilities_2024_2025.csv",
        "final_reference_error_examples_2024_2025.csv",
        "final_reference_confusion_matrix_2024_2025.csv",
        "final_reference_classification_report_2024_2025.csv",
        "final_reference_bootstrap_ci_2024_2025.csv",
        "final_reference_bootstrap_metadata.json",
        "final_reference_baseline_comparison_2024_2025.csv",
        "final_paired_bootstrap_2024_2025.csv",
        "final_paired_bootstrap_2024_2025.json",
        "final_model_protocol.md",
    ]
    reference_hashes = {name: sha256_file(tables_dir / name) for name in reference_files}
    training_source = root / "src" / "block_e_modeling.py"
    if not training_source.exists():
        # Unit-test builders may use a minimal temporary root. The production
        # manifest still authenticates the actual training implementation.
        training_source = Path(__file__).with_name("block_e_modeling.py")
    design_audit_source = root / "src" / "validation_design_audit.py"
    if not design_audit_source.exists():
        design_audit_source = Path(__file__).with_name("validation_design_audit.py")
    production_root = root.resolve() == ROOT.resolve()
    design_hashes = design_evidence_hashes(root, require_complete=production_root)
    manifest = {
        "manifest_schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": "retrospective conditional multifatality classification",
        "dataset": {
            "path": str(base_path.relative_to(root)),
            "sha256": sha256_file(base_path),
            "row_count": int(len(design_base) + len(X_test_raw)),
            "raw_source_manifest_path": source_verification["manifest_path"],
            "raw_source_manifest_sha256": source_verification["manifest_sha256"],
            "raw_source_count": source_verification["source_count"],
        },
        "code_and_libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "builder_code_sha256": sha256_file(Path(__file__)),
            "feature_protocol_sha256": sha256_file(root / "src" / "model_protocol.py"),
            "chronology_protocol_sha256": sha256_file(root / "src" / "chronology.py"),
            "training_code_sha256": sha256_file(training_source),
            "design_audit_code_sha256": sha256_file(design_audit_source),
        },
        "splits": split_metadata,
        "feature_count": len(feature_list),
        "architecture": {
            **selection["selected_config"],
            "seed": int(selection["selected_seed"]),
            "weights_frozen": True,
            "architecture_search_rerun": False,
        },
        "calibration": {
            "method": method,
            "validation_policy": CALIBRATION_POLICY,
            "validation_partition": CALIBRATION_THRESHOLD_VALIDATION_PARTITION,
            "threshold_source": "OOF calibrated 2023 validation predictions",
        },
        "thresholds": thresholds,
        "endpoint_open_gate": {
            "design_rows_materialized_before_open": int(len(design_base)),
            "reference_rows_materialized_after_open": int(len(X_test_raw)),
            "reference_period": "2024-2025",
            "frozen_decision_artifact_hashes": frozen_decision_hashes,
        },
        "artifact_hashes": artifact_hashes,
        "selection_artifact_hashes": {
            "model_selection.json": sha256_file(final_dir / "model_selection.json"),
            "model_selection_baseline_validation.csv": sha256_file(tables_dir / "model_selection_baseline_validation.csv"),
            "model_selection_seed_grid_validation.csv": sha256_file(tables_dir / "model_selection_seed_grid_validation.csv"),
            "model_selection_robustness.csv": sha256_file(tables_dir / "model_selection_robustness.csv"),
            "model_learning_curves.csv": sha256_file(tables_dir / "model_learning_curves.csv"),
            "model_generalization_gap.json": sha256_file(tables_dir / "model_generalization_gap.json"),
        },
        "temporal_diagnostic_artifact_hashes": {
            name: sha256_file(tables_dir / name)
            for name in (
                "temporal_nested_fold_roles.csv",
                "temporal_nested_outer_metrics.csv",
                "temporal_nested_diagnostics.json",
            )
        },
        "reference_artifact_hashes": reference_hashes,
        "design_evidence_artifact_hashes": design_hashes,
        "reference_evaluation": {
            "status": "2024-2025 labels were already observed; metrics are historical reference and cannot justify further tuning",
            "used_for_model_selection": False,
            "used_for_calibration_selection": False,
            "metrics": metrics,
            "bootstrap_ci": ci.to_dict(orient="records"),
            "bootstrap_scope": bootstrap_metadata,
            "confirmatory_independence": False,
            "external_or_prospective_untouched_cohort": False,
        },
        "claims": canonical_claims(reference_comparison, paired_summary),
    }
    (final_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class CanonicalModelBundle:
    """Hash-verified runtime loader for the canonical model."""

    def __init__(self, bundle_dir: Path = FINAL_MODEL_DIR, verify_hashes: bool = True) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        if verify_hashes:
            for name, expected in self.manifest["artifact_hashes"].items():
                actual = sha256_file(self.bundle_dir / name)
                if actual != expected:
                    raise ValueError(f"Artifact hash mismatch for {name}: {actual} != {expected}")
        self.model = keras.models.load_model(self.bundle_dir / "model.keras")
        self.scaler = joblib.load(self.bundle_dir / "scaler.joblib")
        self.encoders = joblib.load(self.bundle_dir / "encoders.joblib")
        self.calibrator = joblib.load(self.bundle_dir / "calibrator.joblib")
        self.thresholds = json.loads((self.bundle_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.calibration_method = self.manifest["calibration"]["method"]

    def predict_dataframe(self, raw: pd.DataFrame) -> pd.DataFrame:
        features = transform_features(raw, self.scaler, self.encoders)
        raw_probability = self.model.predict(features, verbose=0).reshape(-1)
        calibrated_probability = apply_calibrator(self.calibrator, self.calibration_method, raw_probability)
        raw_threshold = float(self.thresholds["raw"]["value"])
        calibrated_threshold = float(self.thresholds["calibrated"]["value"])
        output = pd.DataFrame(
            {
                "raw_probability": raw_probability,
                "calibrated_probability": calibrated_probability,
                "raw_prediction": (raw_probability >= raw_threshold).astype("int8"),
                "calibrated_prediction": (calibrated_probability >= calibrated_threshold).astype("int8"),
            },
            index=raw.index,
        )
        if not np.isfinite(output.to_numpy()).all():
            raise RuntimeError("Canonical inference produced non-finite outputs.")
        return output


if __name__ == "__main__":
    result = build_canonical_bundle()
    print(json.dumps({"model_version": result["model_version"], "metrics": result["reference_evaluation"]["metrics"]}, ensure_ascii=False, indent=2))
