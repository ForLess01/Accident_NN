"""Paired bootstrap comparison between the frozen MLP and the learned baselines.

Diagnostic on the already-observed 2024-2025 reference period: it quantifies
the uncertainty of the metric differences without retraining, reselecting, or
recalibrating anything. MLP probabilities come from the frozen canonical
predictions; each baseline is refit with the exact recipe used for
final_reference_baseline_comparison_2024_2025.csv and validated against it.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_protocol import fit_preprocessor, split_chronological, transform_features

PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "report" / "tables"

SEED = 42
BOOTSTRAP_ITERATIONS = 2000
MATCH_TOLERANCE = 1e-6
METRIC_NAMES = ["pr_auc", "roc_auc", "f1_multifatal"]
COMPARISON_FAMILY_SIZE = 6


def _baseline_models() -> dict[str, object]:
    """Exactly the recipes used by final_model_bundle._baseline_reference."""
    return {
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


def _load_reference_frames() -> dict[str, object]:
    base = pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet")
    splits = split_chronological(base)
    scaler, encoders = fit_preprocessor(splits["X_train_raw"])  # type: ignore[arg-type]
    X_train = transform_features(splits["X_train_raw"], scaler, encoders).astype("float64")  # type: ignore[arg-type]
    X_test = transform_features(splits["X_test_raw"], scaler, encoders).astype("float64")  # type: ignore[arg-type]
    y_train = np.asarray(splits["y_train"], dtype=int)
    y_test = np.asarray(splits["y_test"], dtype=int)

    frozen = pd.read_csv(TABLES_DIR / "final_reference_probabilities_2024_2025.csv")
    expected_index = np.asarray(splits["X_test_raw"].index, dtype=int)  # type: ignore[union-attr]
    if not np.array_equal(frozen["row_index"].to_numpy(dtype=int), expected_index):
        raise RuntimeError("Frozen reference predictions do not align with the chronological split.")
    if not np.array_equal(frozen["actual_multifatal"].to_numpy(dtype=int), y_test):
        raise RuntimeError("Frozen reference labels do not match the rebuilt reference labels.")

    validation = pd.read_csv(TABLES_DIR / "model_selection_baseline_validation.csv").set_index("model")
    published = pd.read_csv(TABLES_DIR / "final_reference_baseline_comparison_2024_2025.csv").set_index("model")

    baselines: dict[str, dict[str, float | np.ndarray]] = {}
    for name, model in _baseline_models().items():
        model.fit(X_train, y_train)  # type: ignore[attr-defined]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
            probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]  # type: ignore[attr-defined]
        threshold = float(validation.loc[name, "threshold"])
        recomputed_pr = average_precision_score(y_test, probabilities)
        recomputed_f1 = f1_score(y_test, (probabilities >= threshold).astype(int))
        for metric_name, recomputed in (("pr_auc", recomputed_pr), ("f1_multifatal", recomputed_f1)):
            expected = float(published.loc[name, metric_name])
            if abs(recomputed - expected) > MATCH_TOLERANCE:
                raise RuntimeError(
                    f"Refit {name} {metric_name} {recomputed:.6f} does not reproduce the published {expected:.6f}."
                )
        baselines[name] = {"probabilities": probabilities, "threshold": threshold}

    return {
        "y_test": y_test,
        "mlp_probabilities": frozen["raw_probability"].to_numpy(dtype=float),
        "mlp_threshold": float(frozen["raw_threshold"].iloc[0]),
        "baselines": baselines,
    }


def _metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "f1_multifatal": f1_score(y, (probabilities >= threshold).astype(int)),
    }


def run_paired_bootstrap() -> dict[str, object]:
    frames = _load_reference_frames()
    y_test = frames["y_test"]
    mlp_prob = frames["mlp_probabilities"]
    mlp_threshold = frames["mlp_threshold"]
    n = len(y_test)

    rows = []
    summary_results: dict[str, dict[str, object]] = {}
    for baseline_name, payload in frames["baselines"].items():
        baseline_prob = payload["probabilities"]
        baseline_threshold = float(payload["threshold"])
        point_mlp = _metrics(y_test, mlp_prob, mlp_threshold)
        point_baseline = _metrics(y_test, baseline_prob, baseline_threshold)

        rng = np.random.default_rng(SEED)
        deltas: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = rng.integers(0, n, size=n)
            y_sample = y_test[sample]
            if y_sample.sum() == 0 or y_sample.sum() == n:
                continue
            resampled_mlp = _metrics(y_sample, mlp_prob[sample], mlp_threshold)
            resampled_baseline = _metrics(y_sample, baseline_prob[sample], baseline_threshold)
            for name in METRIC_NAMES:
                deltas[name].append(resampled_mlp[name] - resampled_baseline[name])

        for name in METRIC_NAMES:
            draws = np.asarray(deltas[name])
            low, high = np.percentile(draws, [2.5, 97.5])
            rows.append(
                {
                    "baseline": baseline_name,
                    "metric": name,
                    "mlp": point_mlp[name],
                    "baseline_value": point_baseline[name],
                    "delta_mlp_minus_baseline": point_mlp[name] - point_baseline[name],
                    "delta_ci95_low": float(low),
                    "delta_ci95_high": float(high),
                    "prob_mlp_better": float((draws > 0).mean()),
                    "nominal_significant_at_5pct": bool(low > 0 or high < 0),
                    "bootstrap_samples": int(len(draws)),
                }
            )
        summary_results[baseline_name] = {
            row["metric"]: row for row in rows if row["baseline"] == baseline_name
        }

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "final_paired_bootstrap_2024_2025.csv", index=False)

    summary = {
        "comparison": "frozen MLP raw scores vs refit baselines, validation-selected thresholds",
        "period": "2024-2025 historical reference (already observed; diagnostic only)",
        "method": "row-level paired percentile bootstrap conditional on frozen predictions",
        "pipeline_refit_per_resample": False,
        "included_uncertainty": "paired row sampling variation within the already-observed historical reference",
        "excluded_uncertainty": [
            "training and model-selection uncertainty",
            "threshold-selection uncertainty",
            "temporal and spatial dependence",
            "repeated consultation of the reference period",
            "future or external generalization",
        ],
        "seed": SEED,
        "iterations": BOOTSTRAP_ITERATIONS,
        "comparison_family_size": COMPARISON_FAMILY_SIZE,
        "interval_coverage": "nominal 95% per comparison",
        "multiplicity_adjustment": "none",
        "simultaneous_familywise_coverage": False,
        "results": summary_results,
    }
    (TABLES_DIR / "final_paired_bootstrap_2024_2025.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_paired_bootstrap(), ensure_ascii=False, indent=2))
