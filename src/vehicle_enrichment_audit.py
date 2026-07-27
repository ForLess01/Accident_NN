"""Validation-only audit of an enriched VEHICULOS contract.

The canonical pipeline consumes three columns of the VEHICULOS workbook
(CÓDIGO SINIESTRO, VEHÍCULO, MODALIDAD DE TRANSPORTE) and collapses them into
seven counts. The workbook publishes twenty-five columns. Several of the unused
ones describe pre-impact attributes of the units involved --- what they carried,
the territorial scope of the service, and the regulatory status of insurance,
technical inspection and operating authorisation --- which are candidate proxies
for occupancy and vehicle condition, the physical determinants absent from the
crash record.

This audit answers two questions in order. First, whether those aggregates
reconstruct the outcome, as the PERSONAS family did; an enrichment that leaks is
rejected regardless of its accuracy. Second, and only if the leakage test
passes, whether adding them improves discrimination under the frozen protocol.

The historical 2024--2025 reference is never opened here. The canonical bundle
is a read-only input and is never modified.
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
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_e_modeling import choose_threshold, class_weights, evaluate, set_global_seed
from src.model_protocol import EXCLUDED_COLUMNS, transform_features
from src.source_provenance import verify_raw_sources

BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
VEHICLES_PATH = ROOT / "data" / "raw" / "BBDD_ONSV_VEHICULOS_2021-2025.xlsx"
FINAL_DIR = ROOT / "models" / "final"
TABLES_DIR = ROOT / "report" / "tables"

COMPANION_HEADER_ROW = 4
SEEDS = (42, 314, 2718)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260728
DROPOUT, L2, LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS = 0.35, 3e-4, 5e-4, 64, 180

# Post-event or duplicated fields are refused before any measurement: a vehicle
# that fled is known after the crash, and TIPO SINIESTRO restates CLASE.
REFUSED_COLUMNS = ("SITUACIÓN VEHÍCULO", "TIPO SINIESTRO")


def load_vehicles() -> pd.DataFrame:
    verify_raw_sources()
    frame = pd.read_excel(VEHICLES_PATH, sheet_name=0, header=COMPANION_HEADER_ROW, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def enriched_aggregates(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Per-crash aggregates of pre-impact vehicle attributes.

    Every aggregate counts vehicles, never persons, and is bounded above by the
    number of units involved, which the canonical contract already contains.
    """
    frame = vehicles.copy()
    frame["code"] = frame["CÓDIGO SINIESTRO"].astype("string").str.strip()
    frame = frame.dropna(subset=["code"])

    def norm(column: str) -> pd.Series:
        return frame[column].fillna("").astype(str).str.strip().str.upper()

    elemento, ambito = norm("ELEMENTO TRANSPORTADO"), norm("AMBITO SERVICIO")
    soat, citv = norm("ESTADO SOAT"), norm("ESTADO CITV")
    modalidad_estado = norm("ESTADO MODALIDAD")

    frame["_lleva_personas"] = elemento.eq("PERSONAS").astype(int)
    frame["_lleva_carga"] = elemento.eq("CARGA Y/O MERCANCÍAS").astype(int)
    frame["_elemento_desconocido"] = elemento.eq("").astype(int)
    frame["_ambito_larga_distancia"] = ambito.isin(["NACIONAL", "INTERNACIONAL"]).astype(int)
    frame["_ambito_regional"] = ambito.eq("REGIONAL").astype(int)
    frame["_ambito_local"] = ambito.isin(["PROVINCIAL", "DISTRITAL"]).astype(int)
    frame["_ambito_desconocido"] = (ambito.eq("") | ambito.eq("NO SE CONOCE")).astype(int)
    frame["_soat_irregular"] = soat.isin(["VENCIDO", "NO REGISTRA"]).astype(int)
    frame["_citv_irregular"] = citv.isin(["VENCIDO", "NO REGISTRA"]).astype(int)
    frame["_no_habilitado"] = modalidad_estado.eq("NO HABILITADO").astype(int)

    columns = [c for c in frame.columns if c.startswith("_")]
    aggregates = frame.groupby("code")[columns].sum()
    aggregates.columns = [c.lstrip("_") for c in aggregates.columns]
    return aggregates.rename(columns=lambda name: f"veh_{name}")


def leakage_audit(base: pd.DataFrame, aggregates: pd.DataFrame) -> dict[str, Any]:
    """Refuse any aggregate that reproduces the outcome, as PERSONAS did."""
    joined = base.join(aggregates, on="CODIGO_SINIESTRO")
    target = joined["target_multifatal"].astype(int)
    deceased = pd.to_numeric(joined["FALLECIDOS"], errors="coerce")

    findings: list[dict[str, Any]] = []
    for column in aggregates.columns:
        values = joined[column].fillna(0)
        equals_deceased = bool((values == deceased).all())
        # Perfect separation: does any threshold split the target without error?
        separates = False
        for cut in sorted(values.unique()):
            predicted = (values >= cut).astype(int)
            if predicted.equals(target) or (1 - predicted).equals(target):
                separates = True
                break
        findings.append(
            {
                "feature": column,
                "equals_fallecidos_in_all_rows": equals_deceased,
                "perfectly_separates_target": separates,
                "spearman_with_fallecidos": float(values.corr(deceased, method="spearman")),
                "verdict": "reject" if (equals_deceased or separates) else "admissible",
            }
        )
    audit = pd.DataFrame(findings)
    return {
        "table": audit,
        "all_admissible": bool((audit["verdict"] == "admissible").all()),
        "refused_before_measurement": list(REFUSED_COLUMNS),
    }


def build_model(width: int) -> keras.Model:
    model = keras.Sequential([layers.Input((width,))])
    for units in (64, 32):
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(L2)))
        model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def fit_and_score(
    X_fit: np.ndarray, y_fit: pd.Series,
    X_select: np.ndarray, y_select: pd.Series,
    X_train: np.ndarray, y_train: pd.Series,
    X_val: np.ndarray, y_val: pd.Series,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Inner selection on 2021/2022, fixed-epoch refit on 2021--22, score 2023."""
    rows = []
    for seed in SEEDS:
        keras.backend.clear_session()
        set_global_seed(seed)
        model = build_model(X_fit.shape[1])
        history = model.fit(
            X_fit, y_fit, validation_data=(X_select, y_select),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights(y_fit),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=20,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5,
                                                  patience=8, min_lr=1e-5),
            ],
            verbose=0,
        )
        probabilities = model.predict(X_select, verbose=0).reshape(-1)
        rows.append({
            "seed": seed,
            "best_epoch": int(np.argmax(history.history["val_pr_auc"]) + 1),
            "pr_auc_seleccion_2022": float(average_precision_score(y_select, probabilities)),
        })
    grid = pd.DataFrame(rows)
    target = float(grid["pr_auc_seleccion_2022"].median())
    medoid = grid.assign(_d=(grid["pr_auc_seleccion_2022"] - target).abs()).sort_values(["_d", "seed"]).iloc[0]

    keras.backend.clear_session()
    set_global_seed(int(medoid["seed"]))
    final = build_model(X_train.shape[1])
    final.fit(X_train, y_train, epochs=int(medoid["best_epoch"]), batch_size=BATCH_SIZE,
              class_weight=class_weights(y_train), verbose=0)
    probabilities = final.predict(X_val, verbose=0).reshape(-1)
    threshold = float(choose_threshold(y_val, probabilities)["threshold"])
    return probabilities, threshold, grid


def metric_vector(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "f1_multifatal": float(f1_score(y, p >= t, zero_division=0)),
    }


def paired_bootstrap(y, baseline, enriched, t_base, t_rich) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {m: [] for m in ("pr_auc", "roc_auc", "f1_multifatal")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        left, right = metric_vector(y[idx], baseline[idx], t_base), metric_vector(y[idx], enriched[idx], t_rich)
        for metric in samples:
            samples[metric].append(right[metric] - left[metric])
    left, right = metric_vector(y, baseline, t_base), metric_vector(y, enriched, t_rich)
    return pd.DataFrame([
        {
            "comparison": "contrato_enriquecido - contrato_canonico",
            "partition": "validation_2023",
            "metric": metric,
            "canonical_estimate": left[metric],
            "enriched_estimate": right[metric],
            "delta": right[metric] - left[metric],
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
            "prob_enriched_better": float(np.mean(np.asarray(values) > 0)),
            "nominal_significant_at_5pct": bool(
                np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0
            ),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
        }
        for metric, values in samples.items()
    ])


def run() -> dict[str, Any]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PATH, filters=[("FECHA", "<", datetime(2024, 1, 1))])
    base["FECHA"] = pd.to_datetime(base["FECHA"], errors="coerce")
    if int(base["FECHA"].dt.year.max()) > 2023:
        raise RuntimeError("Enrichment audit crossed the frozen 2023 boundary.")

    aggregates = enriched_aggregates(load_vehicles())
    audit = leakage_audit(base, aggregates)
    audit["table"].to_csv(TABLES_DIR / "design_vehicle_enrichment_leakage.csv", index=False)
    if not audit["all_admissible"]:
        raise RuntimeError("An enrichment aggregate reproduces the outcome; contract refused.")

    years = base["FECHA"].dt.year
    scaler, encoders = joblib.load(FINAL_DIR / "scaler.joblib"), joblib.load(FINAL_DIR / "encoders.joblib")

    def canonical_block(mask: pd.Series) -> pd.DataFrame:
        partition = base[mask].drop(columns=[c for c in EXCLUDED_COLUMNS if c in base])
        return transform_features(partition, scaler, encoders)

    def extra_block(mask: pd.Series) -> pd.DataFrame:
        codes = base.loc[mask, "CODIGO_SINIESTRO"]
        return aggregates.reindex(codes).fillna(0).reset_index(drop=True)

    masks = {
        "fit": years.eq(2021), "select": years.eq(2022),
        "train": years.isin([2021, 2022]), "val": years.eq(2023),
    }
    labels = {k: base.loc[m, "target_multifatal"].astype(int).reset_index(drop=True) for k, m in masks.items()}
    canonical = {k: canonical_block(m).reset_index(drop=True) for k, m in masks.items()}
    extra = {k: extra_block(m) for k, m in masks.items()}

    extra_scaler = StandardScaler().fit(extra["train"])
    enriched = {
        k: np.hstack([canonical[k].to_numpy("float32"), extra_scaler.transform(extra[k]).astype("float32")])
        for k in masks
    }
    canonical_arrays = {k: canonical[k].to_numpy("float32") for k in masks}

    base_probabilities, base_threshold, base_grid = fit_and_score(
        canonical_arrays["fit"], labels["fit"], canonical_arrays["select"], labels["select"],
        canonical_arrays["train"], labels["train"], canonical_arrays["val"], labels["val"],
    )
    rich_probabilities, rich_threshold, rich_grid = fit_and_score(
        enriched["fit"], labels["fit"], enriched["select"], labels["select"],
        enriched["train"], labels["train"], enriched["val"], labels["val"],
    )

    y_val = labels["val"].to_numpy()
    comparison = pd.DataFrame([
        {"contrato": "canonico_169", "entradas": canonical_arrays["val"].shape[1],
         "umbral_validacion": base_threshold, **evaluate(labels["val"], base_probabilities, base_threshold)},
        {"contrato": "enriquecido_vehiculos", "entradas": enriched["val"].shape[1],
         "umbral_validacion": rich_threshold, **evaluate(labels["val"], rich_probabilities, rich_threshold)},
    ])
    bootstrap = paired_bootstrap(y_val, base_probabilities, rich_probabilities, base_threshold, rich_threshold)

    comparison.to_csv(TABLES_DIR / "design_vehicle_enrichment_comparison.csv", index=False)
    bootstrap.to_csv(TABLES_DIR / "design_vehicle_enrichment_bootstrap.csv", index=False)

    all_ci_include_zero = bool(((bootstrap["ci_2_5"] <= 0) & (bootstrap["ci_97_5"] >= 0)).all())
    result = {
        "schema_version": 1,
        "generated_by": "src/vehicle_enrichment_audit.py",
        "new_aggregates": list(aggregates.columns),
        "refused_before_measurement": list(REFUSED_COLUMNS),
        "leakage_audit_all_admissible": audit["all_admissible"],
        "reference_period_opened": False,
        "canonical_artifacts_modified": False,
        "all_paired_ci_include_zero": all_ci_include_zero,
        "conclusion": (
            "the enriched contract does not show a decisive advantage on 2023"
            if all_ci_include_zero
            else "the enriched contract shows a nominal difference; a new holdout is required before adopting it"
        ),
    }
    (TABLES_DIR / "design_vehicle_enrichment_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
