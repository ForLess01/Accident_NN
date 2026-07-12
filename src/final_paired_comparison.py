"""Paired bootstrap comparison between the frozen MLP and the logistic baseline.

Diagnostic on the already-observed 2024-2025 reference period: it quantifies
the uncertainty of the metric differences without retraining, reselecting, or
recalibrating anything. MLP probabilities come from the frozen canonical
predictions; the logistic baseline is refit with the exact recipe used for
final_reference_baseline_comparison_2024_2025.csv and validated against it.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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


def _load_reference_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    base = pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet")
    splits = split_chronological(base)
    scaler, encoders = fit_preprocessor(splits["X_train_raw"])  # type: ignore[arg-type]
    X_train = transform_features(splits["X_train_raw"], scaler, encoders)  # type: ignore[arg-type]
    X_test = transform_features(splits["X_test_raw"], scaler, encoders)  # type: ignore[arg-type]
    y_train = np.asarray(splits["y_train"], dtype=int)
    y_test = np.asarray(splits["y_test"], dtype=int)

    frozen = pd.read_csv(TABLES_DIR / "final_reference_probabilities_2024_2025.csv")
    expected_index = np.asarray(splits["X_test_raw"].index, dtype=int)  # type: ignore[union-attr]
    if not np.array_equal(frozen["row_index"].to_numpy(dtype=int), expected_index):
        raise RuntimeError("Frozen reference predictions do not align with the chronological split.")
    if not np.array_equal(frozen["actual_multifatal"].to_numpy(dtype=int), y_test):
        raise RuntimeError("Frozen reference labels do not match the rebuilt reference labels.")

    logistic = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="liblinear", random_state=SEED
    ).fit(X_train.astype("float64"), y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
        logistic_probabilities = logistic.predict_proba(X_test.astype("float64"))[
            :, list(logistic.classes_).index(1)
        ]

    mlp_probabilities = frozen["raw_probability"].to_numpy(dtype=float)
    mlp_threshold = float(frozen["raw_threshold"].iloc[0])

    validation = pd.read_csv(TABLES_DIR / "model_selection_baseline_validation.csv").set_index("model")
    logistic_threshold = float(validation.loc["LogisticRegression_balanced", "threshold"])
    return y_test, mlp_probabilities, logistic_probabilities, mlp_threshold, logistic_threshold


def _validate_against_published(
    y_test: np.ndarray, logistic_probabilities: np.ndarray, logistic_threshold: float
) -> None:
    published = (
        pd.read_csv(TABLES_DIR / "final_reference_baseline_comparison_2024_2025.csv")
        .set_index("model")
        .loc["LogisticRegression_balanced"]
    )
    recomputed_pr = average_precision_score(y_test, logistic_probabilities)
    recomputed_f1 = f1_score(y_test, (logistic_probabilities >= logistic_threshold).astype(int))
    if abs(recomputed_pr - float(published["pr_auc"])) > MATCH_TOLERANCE:
        raise RuntimeError(
            f"Refit logistic PR-AUC {recomputed_pr:.6f} does not reproduce the published "
            f"{float(published['pr_auc']):.6f}."
        )
    if abs(recomputed_f1 - float(published["f1_multifatal"])) > MATCH_TOLERANCE:
        raise RuntimeError(
            f"Refit logistic F1 {recomputed_f1:.6f} does not reproduce the published "
            f"{float(published['f1_multifatal']):.6f}."
        )


def run_paired_bootstrap() -> dict[str, object]:
    y_test, mlp_prob, log_prob, mlp_threshold, log_threshold = _load_reference_frames()
    _validate_against_published(y_test, log_prob, log_threshold)

    rng = np.random.default_rng(SEED)
    n = len(y_test)
    metric_names = ["pr_auc", "roc_auc", "f1_multifatal"]

    def metrics_for(y: np.ndarray, mlp_p: np.ndarray, log_p: np.ndarray) -> dict[str, tuple[float, float]]:
        return {
            "pr_auc": (average_precision_score(y, mlp_p), average_precision_score(y, log_p)),
            "roc_auc": (roc_auc_score(y, mlp_p), roc_auc_score(y, log_p)),
            "f1_multifatal": (
                f1_score(y, (mlp_p >= mlp_threshold).astype(int)),
                f1_score(y, (log_p >= log_threshold).astype(int)),
            ),
        }

    point = metrics_for(y_test, mlp_prob, log_prob)
    deltas: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.integers(0, n, size=n)
        y_sample = y_test[sample]
        if y_sample.sum() == 0 or y_sample.sum() == n:
            continue
        resampled = metrics_for(y_sample, mlp_prob[sample], log_prob[sample])
        for name in metric_names:
            deltas[name].append(resampled[name][0] - resampled[name][1])

    rows = []
    for name in metric_names:
        draws = np.asarray(deltas[name])
        low, high = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "metric": name,
                "mlp": point[name][0],
                "logistic": point[name][1],
                "delta_mlp_minus_logistic": point[name][0] - point[name][1],
                "delta_ci95_low": float(low),
                "delta_ci95_high": float(high),
                "prob_mlp_better": float((draws > 0).mean()),
                "significant_at_5pct": bool(low > 0 or high < 0),
                "bootstrap_samples": int(len(draws)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "final_paired_bootstrap_2024_2025.csv", index=False)

    summary = {
        "comparison": "MLP_definitiva (raw, t=%.2f) vs LogisticRegression_balanced (raw, t=%.2f)"
        % (mlp_threshold, log_threshold),
        "period": "2024-2025 historical reference (already observed; diagnostic only)",
        "method": "paired bootstrap over records, percentile CI",
        "seed": SEED,
        "iterations": BOOTSTRAP_ITERATIONS,
        "results": {row["metric"]: row for row in table.to_dict(orient="records")},
    }
    (TABLES_DIR / "final_paired_bootstrap_2024_2025.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_paired_bootstrap(), ensure_ascii=False, indent=2))
