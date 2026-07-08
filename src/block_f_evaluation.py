from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_d_preprocessing import split_dataset
from src.preprocessing import load_artifacts, preparar_entrada


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
SECTIONS_DIR = ROOT / "report" / "sections"
SEED = 42


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["target_mortal"].astype("int8")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["target_mortal"].astype("int8")

    base = pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet")
    raw_splits = split_dataset(base)
    X_test_raw = raw_splits["X_test_raw"].reset_index(drop=True)  # type: ignore[union-attr]
    return X_train, y_train, X_test, y_test, X_test_raw


def metric_row(model_name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "modelo": model_name,
        "threshold": threshold,
        "f1_mortal": f1_score(y_true, predictions, pos_label=1, zero_division=0),
        "recall_mortal": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "precision_mortal": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
    }


def positive_class_probabilities(model: object, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)  # type: ignore[attr-defined]
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def evaluate_baselines_on_test(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float | str]]:
    baselines = [
        ("DummyClassifier_most_frequent", DummyClassifier(strategy="most_frequent")),
        (
            "LogisticRegression_balanced",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED, solver="liblinear"),
        ),
        (
            "RandomForest_balanced",
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        ),
    ]
    rows: list[dict[str, float | str]] = []
    for name, model in baselines:
        model.fit(X_train, y_train)
        probabilities = positive_class_probabilities(model, X_test)
        rows.append(metric_row(name, y_test, probabilities, threshold=0.5))
    return rows


def plot_confusion(y_true: pd.Series, predictions: np.ndarray) -> None:
    matrix_abs = confusion_matrix(y_true, predictions, labels=[0, 1])
    matrix_norm = confusion_matrix(y_true, predictions, labels=[0, 1], normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(matrix_abs, display_labels=["No mortal", "Mortal"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Matriz de confusión")
    ConfusionMatrixDisplay(matrix_norm, display_labels=["No mortal", "Mortal"]).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Normalizada por clase real")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig16_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_curves(y_true: pd.Series, probabilities: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Curva ROC - test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig17_roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Curva Precision-Recall - test")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig18_precision_recall_curve.png", dpi=160)
    plt.close(fig)


def write_classification_report(y_true: pd.Series, predictions: np.ndarray) -> None:
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["no_mortal", "mortal"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(TABLES_DIR / "tab05_classification_report_test.csv")


def analyze_false_negatives(
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    false_negative_mask = (y_test.to_numpy() == 1) & (predictions == 0)
    false_negatives = X_test_raw.loc[false_negative_mask].copy()
    false_negatives["probabilidad_mortal"] = probabilities[false_negative_mask]
    false_negatives = false_negatives.sort_values("probabilidad_mortal", ascending=False)
    sample = false_negatives.head(5)
    sample.to_csv(TABLES_DIR / "tab05_false_negatives_examples.csv", index=False)

    fn_count = int(false_negative_mask.sum())
    total_mortal = int((y_test == 1).sum())
    paragraph = (
        "El modelo final produjo "
        f"{fn_count} falsos negativos sobre {total_mortal} accidentes mortales del test. "
        "En este dominio, un falso negativo significa subestimar un accidente que sí fue mortal; "
        "por eso es el error de mayor costo operativo. "
        f"El umbral seleccionado ({threshold:.2f}) se mantuvo porque en validación ofrecía el mejor equilibrio bajo la regla definida: "
        "maximizar recall entre umbrales con F1 razonable. "
        "Los ejemplos de falsos negativos quedan listados en tab05_false_negatives_examples.csv para discusión cualitativa."
    )
    (SECTIONS_DIR / "resultados_bloque_f.tex").write_text(paragraph + "\n", encoding="utf-8")
    return sample


def run_shap(model: keras.Model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    background = X_train.sample(min(200, len(X_train)), random_state=SEED).to_numpy(dtype="float32")
    shap_input = X_test.sample(min(1000, len(X_test)), random_state=SEED).to_numpy(dtype="float32")
    feature_names = X_test.columns.tolist()

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(shap_input)
    values = shap_values[0] if isinstance(shap_values, list) else shap_values
    values = np.asarray(values).squeeze()

    shap.summary_plot(values, shap_input, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig19_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.head(20).to_csv(TABLES_DIR / "tab08_shap_importance.csv", index=False)
    top5 = importance.head(5)
    top5.to_csv(TABLES_DIR / "tab08_shap_top5.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top5["feature"][::-1], top5["mean_abs_shap"][::-1])
    ax.set_title("Top-5 importancia SHAP media")
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig20_shap_bar.png", dpi=160)
    plt.close(fig)
    return top5


def evaluate_demo_cases(model: keras.Model, threshold: float) -> None:
    scaler, encoders = load_artifacts(MODELS_DIR)
    demo_cases = pd.read_csv(PROCESSED_DIR / "demo_cases.csv")
    features = preparar_entrada(demo_cases, scaler=scaler, encoders=encoders)
    probabilities = model.predict(features, verbose=0).reshape(-1)
    output = demo_cases[["caso_id"]].copy()
    output["probabilidad_mortal"] = probabilities
    output["clasificacion"] = np.where(probabilities >= threshold, "MORTAL", "NO_MORTAL")
    output["threshold"] = threshold
    output["observacion"] = demo_cases["esperado_cualitativo"]
    output.to_csv(TABLES_DIR / "tab06_demo_cases.csv", index=False)


def calibration_diagnostics(y_true: pd.Series, probabilities: np.ndarray) -> None:
    brier = brier_score_loss(y_true, probabilities)
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    table = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    table["brier_score"] = brier
    table.to_csv(TABLES_DIR / "tab07_calibracion.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.plot(prob_pred, prob_true, marker="o", label=f"Modelo (Brier={brier:.4f})")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Curva de calibración - test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig21_calibracion.png", dpi=160)
    plt.close(fig)


def run_block_f() -> dict[str, object]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, X_test_raw = load_splits()
    model = keras.models.load_model(MODELS_DIR / "severidad_nn.keras")
    threshold_data = json.loads((MODELS_DIR / "threshold.json").read_text())
    threshold = float(threshold_data["threshold"])

    probabilities = model.predict(X_test, verbose=0).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)

    plot_confusion(y_test, predictions)
    plot_curves(y_test, probabilities)
    write_classification_report(y_test, predictions)

    baseline_rows = evaluate_baselines_on_test(X_train, y_train, X_test, y_test)
    mlp_row = metric_row("MLP_R5_one_hidden_layer", y_test, probabilities, threshold=threshold)
    pd.DataFrame([*baseline_rows, mlp_row]).to_csv(TABLES_DIR / "tab03_model_comparison_test.csv", index=False)

    false_negative_sample = analyze_false_negatives(X_test_raw, y_test, probabilities, predictions, threshold)
    top5 = run_shap(model, X_train, X_test)
    evaluate_demo_cases(model, threshold)
    calibration_diagnostics(y_test, probabilities)

    summary = {
        "test_rows": int(len(y_test)),
        "threshold": threshold,
        "f1_mortal": float(mlp_row["f1_mortal"]),
        "recall_mortal": float(mlp_row["recall_mortal"]),
        "precision_mortal": float(mlp_row["precision_mortal"]),
        "pr_auc": float(mlp_row["pr_auc"]),
        "roc_auc": float(mlp_row["roc_auc"]),
        "accuracy": float(mlp_row["accuracy"]),
        "false_negatives": int(((y_test.to_numpy() == 1) & (predictions == 0)).sum()),
        "false_negative_examples_written": int(false_negative_sample.shape[0]),
        "top5_shap": top5["feature"].tolist(),
        "test_evaluations": 1,
    }
    (TABLES_DIR / "tab05_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_block_f(), ensure_ascii=False, indent=2))

