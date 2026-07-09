from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
SECTIONS_DIR = ROOT / "report" / "sections"
SEED = 42


def load_validation_and_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_val = pd.read_parquet(PROCESSED_DIR / "X_val.parquet")
    y_val = pd.read_parquet(PROCESSED_DIR / "y_val.parquet")["target_multifatal"].astype("int8")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["target_multifatal"].astype("int8")
    return X_val, y_val, X_test, y_test


def expected_calibration_error(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        weight = mask.mean()
        ece += weight * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def fit_calibrators(y_val: pd.Series, raw_val: np.ndarray) -> dict[str, object]:
    platt = LogisticRegression(solver="lbfgs", random_state=SEED)
    platt.fit(raw_val.reshape(-1, 1), y_val)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_val, y_val)

    return {"platt": platt, "isotonic": isotonic}


def apply_calibrator(method: str, calibrator: object, probabilities: np.ndarray) -> np.ndarray:
    if method == "platt":
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]  # type: ignore[attr-defined]
    if method == "isotonic":
        return calibrator.predict(probabilities)  # type: ignore[attr-defined]
    return probabilities


def calibration_metrics(dataset: str, method: str, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
    return {
        "dataset": dataset,
        "calibrador": method,
        "brier_score": brier_score_loss(y_true, probabilities),
        "ece_10_bins": expected_calibration_error(y_true, probabilities, n_bins=10),
        "promedio_score": float(np.mean(probabilities)),
        "prevalencia_observada": float(np.mean(y_true)),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def select_calibrator(rows: list[dict[str, float | str]]) -> str:
    validation_rows = [row for row in rows if row["dataset"] == "validation" and row["calibrador"] != "raw_sigmoid"]
    return str(min(validation_rows, key=lambda row: (float(row["brier_score"]), str(row["calibrador"])))["calibrador"])


def plot_posthoc_calibration(y_test: pd.Series, raw_test: np.ndarray, calibrated_test: np.ndarray, selected_method: str) -> None:
    raw_true, raw_pred = calibration_curve(y_test, raw_test, n_bins=8, strategy="quantile")
    cal_true, cal_pred = calibration_curve(y_test, calibrated_test, n_bins=8, strategy="quantile")

    raw_brier = brier_score_loss(y_test, raw_test)
    calibrated_brier = brier_score_loss(y_test, calibrated_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.plot(raw_pred, raw_true, marker="o", label=f"Sigmoide cruda (Brier={raw_brier:.4f})")
    ax.plot(
        cal_pred,
        cal_true,
        marker="s",
        label=f"Post-hoc {selected_method} (Brier={calibrated_brier:.4f})",
    )
    ax.set_xlabel("Score medio predicho")
    ax.set_ylabel("Frecuencia mortal observada")
    ax.set_title("Calibración post-hoc - test congelado")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig22_calibracion_posthoc.png", dpi=160)
    plt.close(fig)


def bootstrap_confidence_intervals(
    y_true: pd.Series,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = y_true.to_numpy()
    n = len(y)

    def metrics_for_index(index: np.ndarray) -> dict[str, float]:
        y_sample = y[index]
        raw_sample = raw_probabilities[index]
        calibrated_sample = calibrated_probabilities[index]
        pred_sample = (raw_sample >= threshold).astype(int)
        return {
            "f1_multifatal": f1_score(y_sample, pred_sample, pos_label=1, zero_division=0),
            "recall_multifatal": recall_score(y_sample, pred_sample, pos_label=1, zero_division=0),
            "precision_multifatal": precision_score(y_sample, pred_sample, pos_label=1, zero_division=0),
            "accuracy": accuracy_score(y_sample, pred_sample),
            "pr_auc": average_precision_score(y_sample, raw_sample),
            "roc_auc": roc_auc_score(y_sample, raw_sample),
            "brier_raw": brier_score_loss(y_sample, raw_sample),
            "brier_calibrated": brier_score_loss(y_sample, calibrated_sample),
        }

    estimates = metrics_for_index(np.arange(n))
    samples: dict[str, list[float]] = {metric: [] for metric in estimates}
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        if len(np.unique(y[index])) < 2:
            continue
        sample_metrics = metrics_for_index(index)
        for metric, value in sample_metrics.items():
            samples[metric].append(float(value))

    rows = []
    for metric, estimate in estimates.items():
        values = np.asarray(samples[metric])
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_samples": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def write_report_snippet(
    selected_method: str,
    table: pd.DataFrame,
    ci_table: pd.DataFrame,
) -> None:
    raw_test = table[(table["dataset"] == "test") & (table["calibrador"] == "raw_sigmoid")].iloc[0]
    calibrated_test = table[(table["dataset"] == "test") & (table["calibrador"] == selected_method)].iloc[0]
    f1 = ci_table[ci_table["metric"] == "f1_multifatal"].iloc[0]
    recall = ci_table[ci_table["metric"] == "recall_multifatal"].iloc[0]
    pr_auc = ci_table[ci_table["metric"] == "pr_auc"].iloc[0]

    text = f"""Para corregir la interpretación de la salida sigmoide, se ajustó un calibrador post-hoc usando solamente el conjunto de validación. Se compararon Platt e isotónica por \\emph{{Brier score}} de validación y se seleccionó \\textbf{{{selected_method}}}. El test se mantuvo como diagnóstico congelado: no se usó para elegir el calibrador.

En test, la salida sigmoide cruda obtuvo Brier {float(raw_test['brier_score']):.4f} y ECE {float(raw_test['ece_10_bins']):.4f}; el score calibrado obtuvo Brier {float(calibrated_test['brier_score']):.4f} y ECE {float(calibrated_test['ece_10_bins']):.4f}. Por eso, la GUI deja de presentar la salida como ``probabilidad operacional'' y la muestra como \\textbf{{score calibrado de riesgo mortal}}.

Además, se calcularon intervalos de confianza bootstrap al 95\\,\\% sobre las predicciones congeladas del test: F1-mortal {float(f1['estimate']):.4f} [{float(f1['ci95_low']):.4f}, {float(f1['ci95_high']):.4f}], recall-mortal {float(recall['estimate']):.4f} [{float(recall['ci95_low']):.4f}, {float(recall['ci95_high']):.4f}] y PR-AUC {float(pr_auc['estimate']):.4f} [{float(pr_auc['ci95_low']):.4f}, {float(pr_auc['ci95_high']):.4f}]. Estos intervalos evitan vender el resultado como un punto exacto: muestran la incertidumbre real de un test pequeño y desbalanceado.
"""
    (SECTIONS_DIR / "calibration_posthoc.tex").write_text(text, encoding="utf-8")


def run_posthoc_calibration() -> dict[str, object]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_val, y_val, X_test, y_test = load_validation_and_test()
    model = keras.models.load_model(MODELS_DIR / "letalidad_nn.keras")
    threshold = float(json.loads((MODELS_DIR / "threshold.json").read_text(encoding="utf-8"))["threshold"])

    raw_val = model.predict(X_val, verbose=0).reshape(-1)
    raw_test = model.predict(X_test, verbose=0).reshape(-1)

    calibrators = fit_calibrators(y_val, raw_val)

    rows: list[dict[str, float | str]] = [
        calibration_metrics("validation", "raw_sigmoid", y_val, raw_val),
        calibration_metrics("test", "raw_sigmoid", y_test, raw_test),
    ]
    calibrated_values: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    for method, calibrator in calibrators.items():
        calibrated_values["validation"][method] = apply_calibrator(method, calibrator, raw_val)
        calibrated_values["test"][method] = apply_calibrator(method, calibrator, raw_test)
        rows.append(calibration_metrics("validation", method, y_val, calibrated_values["validation"][method]))
        rows.append(calibration_metrics("test", method, y_test, calibrated_values["test"][method]))

    table = pd.DataFrame(rows).sort_values(["dataset", "calibrador"]).reset_index(drop=True)
    table.to_csv(TABLES_DIR / "tab11_calibration_posthoc.csv", index=False)

    selected_method = select_calibrator(rows)
    selected_calibrator = calibrators[selected_method]
    selected_test = calibrated_values["test"][selected_method]

    calibrator_payload = {
        "version": 1,
        "method": selected_method,
        "model": selected_calibrator,
        "fit_dataset": "validation",
        "raw_validation_brier": float(
            table[(table["dataset"] == "validation") & (table["calibrador"] == "raw_sigmoid")]["brier_score"].iloc[0]
        ),
        "calibrated_validation_brier": float(
            table[(table["dataset"] == "validation") & (table["calibrador"] == selected_method)]["brier_score"].iloc[0]
        ),
    }
    joblib.dump(calibrator_payload, MODELS_DIR / "calibrator.pkl")

    plot_posthoc_calibration(y_test, raw_test, selected_test, selected_method)
    ci_table = bootstrap_confidence_intervals(y_test, raw_test, selected_test, threshold=threshold)
    ci_table.to_csv(TABLES_DIR / "tab12_bootstrap_ci_test.csv", index=False)
    write_report_snippet(selected_method, table, ci_table)

    summary = {
        "selected_calibrator": selected_method,
        "fit_dataset": "validation",
        "raw_test_brier": float(table[(table["dataset"] == "test") & (table["calibrador"] == "raw_sigmoid")]["brier_score"].iloc[0]),
        "calibrated_test_brier": float(table[(table["dataset"] == "test") & (table["calibrador"] == selected_method)]["brier_score"].iloc[0]),
        "raw_test_ece_10_bins": float(table[(table["dataset"] == "test") & (table["calibrador"] == "raw_sigmoid")]["ece_10_bins"].iloc[0]),
        "calibrated_test_ece_10_bins": float(table[(table["dataset"] == "test") & (table["calibrador"] == selected_method)]["ece_10_bins"].iloc[0]),
        "bootstrap_samples": int(ci_table["bootstrap_samples"].min()),
    }
    (TABLES_DIR / "tab11_calibration_posthoc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_posthoc_calibration(), ensure_ascii=False, indent=2))
