from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
SEED = 42


@dataclass(frozen=True)
class GridRun:
    run_id: str
    hidden_units: tuple[int, ...]
    dropouts: tuple[float, ...]
    learning_rate: float
    batch_size: int


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_train_validation() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_val = pd.read_parquet(PROCESSED_DIR / "X_val.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["target_mortal"].astype("int8")
    y_val = pd.read_parquet(PROCESSED_DIR / "y_val.parquet")["target_mortal"].astype("int8")
    return X_train, y_train, X_val, y_val


def evaluate_probabilities(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "modelo": name,
        "threshold": threshold,
        "f1_mortal": f1_score(y_true, predictions, pos_label=1, zero_division=0),
        "precision_mortal": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "recall_mortal": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def positive_class_probabilities(model: object, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)  # type: ignore[attr-defined]
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def train_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> list[dict[str, float | str]]:
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
        probabilities = positive_class_probabilities(model, X_val)
        rows.append(evaluate_probabilities(name, y_val, probabilities))
    return rows


def build_model(n_features: int, run: GridRun) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    for index, units in enumerate(run.hidden_units):
        model.add(layers.Dense(units, activation="relu"))
        if index == 0:
            model.add(layers.BatchNormalization())
        if index < len(run.dropouts):
            model.add(layers.Dropout(run.dropouts[index]))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(run.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def grid_for_architecture_a() -> list[GridRun]:
    return [
        GridRun("R1_base", (32, 16), (0.4, 0.3), 1e-3, 32),
        GridRun("R2_lr_1e_4", (32, 16), (0.4, 0.3), 1e-4, 32),
        GridRun("R3_dropout_plus_0_1", (32, 16), (0.5, 0.4), 1e-3, 32),
        GridRun("R4_half_units", (16, 8), (0.4, 0.3), 1e-3, 32),
        GridRun("R5_one_hidden_layer", (32,), (0.4,), 1e-3, 32),
        GridRun("R6_batch_64", (32, 16), (0.4, 0.3), 1e-3, 64),
    ]


def class_weights(y_train: pd.Series) -> dict[int, float]:
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def train_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[keras.Model, GridRun, list[dict[str, float | int | str]], keras.callbacks.History]:
    grid = grid_for_architecture_a()
    weights = class_weights(y_train)

    best_model: keras.Model | None = None
    best_run: GridRun | None = None
    best_history: keras.callbacks.History | None = None
    best_score = (-1.0, -1.0)
    rows: list[dict[str, float | int | str]] = []

    for run in grid:
        set_global_seed(SEED)
        model = build_model(X_train.shape[1], run)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / f"{run.run_id}.keras"
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc",
                    mode="max",
                    patience=15,
                    restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_pr_auc",
                    mode="max",
                    save_best_only=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                ),
            ]
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=run.batch_size,
                class_weight=weights,
                callbacks=callbacks,
                verbose=0,
            )
            if checkpoint_path.exists():
                model = keras.models.load_model(checkpoint_path)

        probabilities = model.predict(X_val, verbose=0).reshape(-1)
        metrics = evaluate_probabilities(run.run_id, y_val, probabilities)
        row = {
            **metrics,
            "hidden_units": "-".join(str(unit) for unit in run.hidden_units),
            "dropouts": "-".join(f"{dropout:.1f}" for dropout in run.dropouts),
            "learning_rate": run.learning_rate,
            "batch_size": run.batch_size,
            "epochs_ran": len(history.history["loss"]),
            "best_val_pr_auc_history": max(history.history["val_pr_auc"]),
        }
        rows.append(row)

        score = (float(metrics["f1_mortal"]), float(metrics["pr_auc"]))
        if score > best_score:
            best_score = score
            best_model = model
            best_run = run
            best_history = history

    if best_model is None or best_run is None or best_history is None:
        raise RuntimeError("No neural network run completed.")

    return best_model, best_run, rows, best_history


def plot_history(history: keras.callbacks.History) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["pr_auc"], label="train_pr_auc")
    axes[1].plot(history.history["val_pr_auc"], label="val_pr_auc")
    axes[1].set_title("PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_training_curves.png", dpi=160)
    plt.close(fig)


def sweep_thresholds(y_val: pd.Series, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rows: list[dict[str, float]] = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "f1_mortal": f1_score(y_val, predictions, pos_label=1, zero_division=0),
                "precision_mortal": precision_score(y_val, predictions, pos_label=1, zero_division=0),
                "recall_mortal": recall_score(y_val, predictions, pos_label=1, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    max_f1 = float(table["f1_mortal"].max())
    candidates = table[table["f1_mortal"] >= max_f1 * 0.90].sort_values(
        ["recall_mortal", "f1_mortal", "precision_mortal"],
        ascending=[False, False, False],
    )
    selected = candidates.iloc[0].to_dict()
    selected["selection_rule"] = "max recall among thresholds with F1 >= 90% of best validation F1"
    return table, selected


def plot_threshold_sweep(table: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold"], table["f1_mortal"], marker="o", label="F1 mortal")
    ax.plot(table["threshold"], table["precision_mortal"], marker="o", label="Precision mortal")
    ax.plot(table["threshold"], table["recall_mortal"], marker="o", label="Recall mortal")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Validation threshold sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_threshold_sweep.png", dpi=160)
    plt.close(fig)


def run_block_e() -> dict[str, object]:
    set_global_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_train_validation()

    baseline_rows = train_baselines(X_train, y_train, X_val, y_val)
    best_model, best_run, grid_rows, best_history = train_grid(X_train, y_train, X_val, y_val)

    best_probabilities = best_model.predict(X_val, verbose=0).reshape(-1)
    best_metrics = evaluate_probabilities(f"MLP_{best_run.run_id}", y_val, best_probabilities)
    tab03 = pd.DataFrame([*baseline_rows, best_metrics])
    tab03.to_csv(TABLES_DIR / "tab03_model_comparison_validation.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(TABLES_DIR / "tab04_nn_grid_validation.csv", index=False)

    best_model.save(MODELS_DIR / "severidad_nn.keras")
    plot_history(best_history)

    threshold_table, selected_threshold = sweep_thresholds(y_val, best_probabilities)
    threshold_table.to_csv(TABLES_DIR / "tab_umbral_validacion.csv", index=False)
    plot_threshold_sweep(threshold_table)
    (MODELS_DIR / "threshold.json").write_text(
        json.dumps(selected_threshold, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "best_run": best_run.run_id,
        "best_validation_f1_mortal_at_0_5": float(best_metrics["f1_mortal"]),
        "best_validation_pr_auc": float(best_metrics["pr_auc"]),
        "selected_threshold": float(selected_threshold["threshold"]),
        "selected_threshold_recall_mortal": float(selected_threshold["recall_mortal"]),
        "selected_threshold_f1_mortal": float(selected_threshold["f1_mortal"]),
        "grid_runs": len(grid_rows),
        "test_touched": False,
    }
    (TABLES_DIR / "tab04_nn_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_block_e(), ensure_ascii=False, indent=2))
