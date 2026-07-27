"""Validation-only audit of entity embeddings against the canonical one-hot MLP.

The canonical contract feeds 169 columns to the network, of which 153 are binary
one-hot indicators and 48 are hand-crafted categorical crosses. That representation
is the regime where tree ensembles are strongest, which is a candidate explanation
for the observed parity between the canonical MLP and Random Forest.

This audit tests exactly one hypothesis: whether replacing the sparse one-hot
block with learned dense entity embeddings improves discrimination, keeping the
trunk (64-32-1), the optimizer, the seeds, the temporal protocol and the
threshold policy identical. The comparison is therefore attributable to the input
representation alone.

The historical 2024--2025 reference is never read here: the audit reads only
2021--2023 at the Parquet boundary. The canonical bundle is a read-only input and
is never modified.
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_e_modeling import choose_threshold, class_weights, evaluate, set_global_seed
from src.model_protocol import (
    CONTINUOUS_COLUMNS,
    EXCLUDED_COLUMNS,
    derive_base_features,
    transform_features,
)

BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
FINAL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"

SEEDS = (42, 314, 2718)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260727
DROPOUT = 0.35
L2 = 3e-4
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 180

# Categorical fields embedded as dense vectors, with the encoder vocabulary that
# the canonical contract already froze. The two crossed families of the canonical
# contract (road_type_zone, road_network_class) are deliberately absent: the
# hypothesis is that the trunk recovers those interactions from the components.
EMBEDDED_FIELDS = (
    ("DEPARTAMENTO", "departamento_categories"),
    ("via_prefijo", "via_prefijo_categories"),
    ("region_natural", "region_categories"),
    ("franja", "franja_categories"),
    ("CLASE", "clase_categories"),
    ("ZONA", "zona_categories"),
    ("RED_VIAL", "red_vial_categories"),
    ("TIPO_VIA", "tipo_via_categories"),
    ("CLIMA", "clima_categories"),
    ("CARACTERISTICA_VIA", "caracteristica_categories"),
    ("PERFIL_VIA", "perfil_categories"),
    ("SUPERFICIE", "superficie_categories"),
)

# Continuous and already-numeric columns shared verbatim with the canonical model.
# night_rural and rain_curve are excluded for the same reason as the crosses.
NUMERIC_COLUMNS = (
    "mes_sin",
    "mes_cos",
    "dia_semana_sin",
    "dia_semana_cos",
    "fin_de_semana",
    "hora_faltante",
    "hora_sin",
    "hora_cos",
    "nocturno",
    "coord_faltante",
)


def embedding_dimension(cardinality: int) -> int:
    """Conventional square-root rule, capped so no single entity dominates."""
    return int(min(8, max(2, round(cardinality ** 0.5))))


def read_design_period(path: Path = BASE_PATH) -> pd.DataFrame:
    """Read only 2021--2023; the historical reference cannot enter this process."""
    frame = pd.read_parquet(path, filters=[("FECHA", "<", datetime(2024, 1, 1))])
    dates = pd.to_datetime(frame["FECHA"], errors="coerce")
    if dates.isna().any() or frame.empty or int(dates.dt.year.max()) > 2023:
        raise RuntimeError("Embedding audit read crossed the frozen 2023 boundary.")
    return frame


def split_design_period(frame: pd.DataFrame) -> dict[str, Any]:
    """Split into the inner fit/selection years and the 2023 evaluation partition."""
    frame = frame.copy()
    frame["FECHA"] = pd.to_datetime(frame["FECHA"], errors="coerce")
    if frame["FECHA"].isna().any() or "target_multifatal" not in frame:
        raise ValueError("Embedding audit requires valid FECHA and target_multifatal.")
    years = frame["FECHA"].dt.year
    if not years.isin([2021, 2022, 2023]).all():
        raise ValueError("Embedding audit accepts only 2021--2023 rows.")

    def xy(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = partition.drop(columns=[c for c in EXCLUDED_COLUMNS if c in partition])
        return features, partition["target_multifatal"].astype("int8")

    inner_fit, inner_select = xy(frame[years.eq(2021)]), xy(frame[years.eq(2022)])
    train, validation = xy(frame[years.isin([2021, 2022])]), xy(frame[years.eq(2023)])
    if min(len(inner_fit[0]), len(inner_select[0]), len(validation[0])) == 0:
        raise ValueError("Embedding audit requires non-empty 2021, 2022 and 2023 partitions.")
    return {
        "inner_fit": inner_fit,
        "inner_select": inner_select,
        "train": train,
        "validation": validation,
    }


def build_vocabularies(encoders: dict[str, Any]) -> dict[str, list[str]]:
    """Reuse the frozen encoder vocabularies so both models share one category set."""
    return {field: list(encoders[key]) for field, key in EMBEDDED_FIELDS}


def encode_inputs(
    raw: pd.DataFrame,
    scaler: Any,
    encoders: dict[str, Any],
    vocabularies: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Map a raw partition to integer entity codes plus the shared numeric block.

    An unseen or missing category receives the reserved out-of-vocabulary index,
    which mirrors how the canonical contract routes unknown values to DESCONOCIDO.
    """
    base = derive_base_features(raw, encoders["via_frequency_map"])
    base["LATITUD"] = base["LATITUD"].fillna(float(encoders["lat_median"]))
    base["LONGITUD"] = base["LONGITUD"].fillna(float(encoders["lon_median"]))

    inputs: dict[str, np.ndarray] = {}
    for field, _ in EMBEDDED_FIELDS:
        vocabulary = vocabularies[field]
        lookup = {category: index for index, category in enumerate(vocabulary)}
        oov = len(vocabulary)
        codes = base[field].map(lambda value: lookup.get(value, oov))
        inputs[f"entity_{field}"] = codes.to_numpy(dtype="int32").reshape(-1, 1)

    scaled = scaler.transform(base[list(CONTINUOUS_COLUMNS)])
    numeric = np.hstack([scaled, base[list(NUMERIC_COLUMNS)].fillna(0).to_numpy(dtype="float32")])
    inputs["numeric"] = numeric.astype("float32")
    return inputs


def build_embedding_model(vocabularies: dict[str, list[str]], numeric_width: int) -> keras.Model:
    """Entity-embedding front-end feeding the canonical 64-32-1 trunk."""
    entity_inputs, entity_vectors = [], []
    for field, _ in EMBEDDED_FIELDS:
        cardinality = len(vocabularies[field]) + 1  # reserved out-of-vocabulary slot
        dimension = embedding_dimension(cardinality)
        entity_input = keras.Input(shape=(1,), dtype="int32", name=f"entity_{field}")
        vector = layers.Embedding(
            input_dim=cardinality,
            output_dim=dimension,
            embeddings_regularizer=regularizers.l2(L2),
            name=f"emb_{field}",
        )(entity_input)
        entity_inputs.append(entity_input)
        entity_vectors.append(layers.Flatten()(vector))

    numeric_input = keras.Input(shape=(numeric_width,), name="numeric")
    joined = layers.Concatenate(name="representacion_densa")([*entity_vectors, numeric_input])
    hidden = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2))(joined)
    hidden = layers.Dropout(DROPOUT)(hidden)
    hidden = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(L2))(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)
    output = layers.Dense(1, activation="sigmoid")(hidden)

    model = keras.Model([*entity_inputs, numeric_input], output, name="MLP_embeddings_64_32")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def select_epochs(
    fit_inputs: dict[str, np.ndarray],
    y_fit: pd.Series,
    select_inputs: dict[str, np.ndarray],
    y_select: pd.Series,
    vocabularies: dict[str, list[str]],
) -> tuple[pd.DataFrame, int, int]:
    """Mirror the canonical inner selection: fit 2021, select 2022, keep the medoid."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        keras.backend.clear_session()
        set_global_seed(seed)
        model = build_embedding_model(vocabularies, fit_inputs["numeric"].shape[1])
        history = model.fit(
            fit_inputs,
            y_fit,
            validation_data=(select_inputs, y_select),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights(y_fit),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc", mode="max", patience=20, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_pr_auc", mode="max", factor=0.5, patience=8, min_lr=1e-5
                ),
            ],
            verbose=0,
        )
        probabilities = model.predict(select_inputs, verbose=0).reshape(-1)
        threshold = choose_threshold(y_select, probabilities)
        rows.append(
            {
                "seed": seed,
                "epochs_ran": len(history.history["loss"]),
                "best_epoch": int(np.argmax(history.history["val_pr_auc"]) + 1),
                "trainable_parameters": int(model.count_params()),
                "selected_threshold": float(threshold["threshold"]),
                **evaluate(y_select, probabilities, float(threshold["threshold"])),
            }
        )
    grid = pd.DataFrame(rows)
    target = float(grid["pr_auc"].median())
    medoid = grid.assign(_d=(grid["pr_auc"] - target).abs()).sort_values(["_d", "seed"]).iloc[0]
    return grid, int(medoid["seed"]), int(medoid["best_epoch"])


def metric_vector(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "f1_multifatal": float(f1_score(y, prediction, zero_division=0)),
    }


def paired_bootstrap(
    y: np.ndarray,
    canonical: np.ndarray,
    embedding: np.ndarray,
    canonical_threshold: float,
    embedding_threshold: float,
) -> pd.DataFrame:
    """Percentile bootstrap of the paired difference, predictions frozen."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {m: [] for m in ("pr_auc", "roc_auc", "f1_multifatal")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        left = metric_vector(y[indices], canonical[indices], canonical_threshold)
        right = metric_vector(y[indices], embedding[indices], embedding_threshold)
        for metric in samples:
            samples[metric].append(right[metric] - left[metric])
    left_estimates = metric_vector(y, canonical, canonical_threshold)
    right_estimates = metric_vector(y, embedding, embedding_threshold)
    return pd.DataFrame(
        [
            {
                "comparison": "embeddings - one_hot_canonico",
                "partition": "validation_2023",
                "metric": metric,
                "one_hot_estimate": left_estimates[metric],
                "embedding_estimate": right_estimates[metric],
                "delta": right_estimates[metric] - left_estimates[metric],
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "prob_embedding_better": float(np.mean(np.asarray(values) > 0)),
                "nominal_significant_at_5pct": bool(
                    np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0
                ),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED,
            }
            for metric, values in samples.items()
        ]
    )


def representation_inventory(X_train_onehot: pd.DataFrame, numeric_width: int, vocabularies: dict[str, list[str]]) -> pd.DataFrame:
    """Quantify the representation change that the audit is testing."""
    binary = [c for c in X_train_onehot.columns if set(np.unique(X_train_onehot[c].values)) <= {0.0, 1.0}]
    active = (X_train_onehot[binary] != 0).sum(axis=0)
    embedding_width = sum(
        embedding_dimension(len(vocabularies[field]) + 1) for field, _ in EMBEDDED_FIELDS
    )
    return pd.DataFrame(
        [
            {"representation": "one-hot canónica", "input_width": X_train_onehot.shape[1],
             "binary_columns": len(binary),
             "columns_active_in_fewer_than_30_rows": int((active < 30).sum()),
             "columns_always_zero_in_training": int((active == 0).sum())},
            {"representation": "embeddings de entidad", "input_width": embedding_width + numeric_width,
             "binary_columns": 0,
             "columns_active_in_fewer_than_30_rows": 0,
             "columns_always_zero_in_training": 0},
        ]
    )


def figure(comparison: pd.DataFrame, bootstrap: pd.DataFrame, inventory: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    palette = ["#496B7C", "#A63F12"]

    metrics = ["pr_auc", "roc_auc", "f1_multifatal"]
    labels = ["PR-AUC", "ROC-AUC", "F1"]
    width = 0.36
    positions = np.arange(len(metrics))
    onehot = [float(comparison.loc[comparison.model.eq("one_hot_canonico"), m].iloc[0]) for m in metrics]
    embed = [float(comparison.loc[comparison.model.eq("embeddings"), m].iloc[0]) for m in metrics]
    axes[0].bar(positions - width / 2, onehot, width, label="One-hot canónica", color=palette[0])
    axes[0].bar(positions + width / 2, embed, width, label="Embeddings", color=palette[1])
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Validación 2023")
    axes[0].set_title("Desempeño por representación")
    axes[0].legend(fontsize=8)

    ordered = bootstrap.set_index("metric").loc[metrics].reset_index()
    y = np.arange(len(ordered))
    axes[1].errorbar(
        ordered["delta"], y,
        xerr=[ordered["delta"] - ordered["ci_2_5"], ordered["ci_97_5"] - ordered["delta"]],
        fmt="D", color="#A63F12", ecolor="#496B7C", capsize=4,
    )
    axes[1].axvline(0, color="#202522", linewidth=1)
    axes[1].set_yticks(y, labels)
    axes[1].set_title("Embeddings − one-hot · bootstrap pareado 95 %")
    axes[1].set_xlabel("Diferencia en validación 2023")

    axes[2].bar(
        ["One-hot\ncanónica", "Embeddings"],
        inventory["input_width"],
        color=palette,
    )
    for index, row in inventory.reset_index(drop=True).iterrows():
        axes[2].text(index, row["input_width"], f"{int(row['input_width'])}", ha="center", va="bottom")
    axes[2].set_ylabel("Ancho del vector de entrada")
    axes[2].set_title("Dimensionalidad de la representación")

    fig.suptitle(
        "Auditoría de representación: embeddings de entidad vs. one-hot · validación 2023",
        fontsize=15, fontweight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Tronco 64-32-1, optimizador, semillas, protocolo temporal y política de umbral idénticos. "
        "La referencia 2024–2025 no se abre en esta auditoría.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(FIGURES_DIR / "design_embedding_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> dict[str, Any]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    splits = split_design_period(read_design_period(BASE_PATH))
    scaler = joblib.load(FINAL_DIR / "scaler.joblib")
    encoders = joblib.load(FINAL_DIR / "encoders.joblib")
    vocabularies = build_vocabularies(encoders)

    (X_fit_raw, y_fit) = splits["inner_fit"]
    (X_select_raw, y_select) = splits["inner_select"]
    (X_train_raw, y_train) = splits["train"]
    (X_val_raw, y_val) = splits["validation"]

    fit_inputs = encode_inputs(X_fit_raw, scaler, encoders, vocabularies)
    select_inputs = encode_inputs(X_select_raw, scaler, encoders, vocabularies)
    train_inputs = encode_inputs(X_train_raw, scaler, encoders, vocabularies)
    val_inputs = encode_inputs(X_val_raw, scaler, encoders, vocabularies)

    grid, selected_seed, fixed_epochs = select_epochs(
        fit_inputs, y_fit, select_inputs, y_select, vocabularies
    )

    keras.backend.clear_session()
    set_global_seed(selected_seed)
    model = build_embedding_model(vocabularies, train_inputs["numeric"].shape[1])
    model.fit(
        train_inputs,
        y_train,
        epochs=fixed_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights(y_train),
        verbose=0,
    )
    embedding_probabilities = model.predict(val_inputs, verbose=0).reshape(-1)
    embedding_threshold = float(choose_threshold(y_val, embedding_probabilities)["threshold"])

    canonical = keras.models.load_model(FINAL_DIR / "model.keras")
    X_val_onehot = transform_features(X_val_raw, scaler, encoders)
    canonical_probabilities = canonical.predict(X_val_onehot, verbose=0).reshape(-1)
    canonical_threshold = float(choose_threshold(y_val, canonical_probabilities)["threshold"])

    comparison = pd.DataFrame(
        [
            {
                "model": "one_hot_canonico",
                "input_width": int(X_val_onehot.shape[1]),
                "trainable_parameters": int(canonical.count_params()),
                "seed": 42,
                "threshold_validation": canonical_threshold,
                **evaluate(y_val, canonical_probabilities, canonical_threshold),
            },
            {
                "model": "embeddings",
                "input_width": int(
                    sum(embedding_dimension(len(vocabularies[f]) + 1) for f, _ in EMBEDDED_FIELDS)
                    + train_inputs["numeric"].shape[1]
                ),
                "trainable_parameters": int(model.count_params()),
                "seed": selected_seed,
                "threshold_validation": embedding_threshold,
                **evaluate(y_val, embedding_probabilities, embedding_threshold),
            },
        ]
    )

    bootstrap = paired_bootstrap(
        np.asarray(y_val, dtype=int),
        canonical_probabilities,
        embedding_probabilities,
        canonical_threshold,
        embedding_threshold,
    )

    X_train_onehot = transform_features(X_train_raw, scaler, encoders)
    inventory = representation_inventory(
        X_train_onehot, train_inputs["numeric"].shape[1], vocabularies
    )

    grid.to_csv(TABLES_DIR / "design_embedding_seed_grid.csv", index=False)
    comparison.to_csv(TABLES_DIR / "design_embedding_comparison.csv", index=False)
    bootstrap.to_csv(TABLES_DIR / "design_embedding_bootstrap.csv", index=False)
    inventory.to_csv(TABLES_DIR / "design_embedding_representation.csv", index=False)
    figure(comparison, bootstrap, inventory)

    all_ci_include_zero = bool(((bootstrap["ci_2_5"] <= 0) & (bootstrap["ci_97_5"] >= 0)).all())
    result = {
        "schema_version": 1,
        "generated_by": "src/embedding_design_audit.py",
        "hypothesis": (
            "dense entity embeddings improve discrimination over the sparse one-hot contract "
            "with an identical 64-32-1 trunk"
        ),
        "controlled": ["trunk", "optimizer", "learning_rate", "batch_size", "seeds",
                       "temporal protocol", "threshold policy", "category vocabularies"],
        "changed": ["input representation only"],
        "selected_seed": selected_seed,
        "fixed_refit_epochs_from_inner_selection": fixed_epochs,
        "reference_period_opened": False,
        "canonical_artifacts_modified": False,
        "all_paired_ci_include_zero": all_ci_include_zero,
        "conclusion": (
            "retain the canonical one-hot MLP; embeddings do not show a decisive advantage"
            if all_ci_include_zero
            else "embeddings show a nominal difference; a new holdout is required before adopting them"
        ),
    }
    (TABLES_DIR / "design_embedding_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
