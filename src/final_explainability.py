"""Selection-period global explainability for the definitive MLP.

The generator deliberately reads only pre-2024 raw fields.  Training rows are
used solely as the Gradient SHAP background and 2023 validation rows are the
only examples explained.  It never reads endpoint labels, selects a model,
changes calibration, or writes into the canonical runtime bundle except for
adding hashes and provenance references to its manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/accident_nn_matplotlib")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_protocol import transform_features


SEED = 20260709
BACKGROUND_SIZE = 256
EXPLANATION_SIZE = 512
SHAP_NSAMPLES = 200

# Reading an explicit projection plus a Parquet predicate makes the temporal
# boundary auditable: neither endpoint rows nor any outcome column enter this
# process, even transiently.
EXPLANATION_RAW_COLUMNS = [
    "FECHA",
    "HORA",
    "hora_entera",
    "DEPARTAMENTO",
    "CODIGO_VIA",
    "LATITUD",
    "LONGITUD",
    "CLASE",
    "ZONA",
    "RED_VIAL",
    "TIPO_VIA",
    "CLIMA",
    "CARACTERISTICA_VIA",
    "PERFIL_VIA",
    "SUPERFICIE",
    # v2 scene aggregates; per-person outcome columns never enter this list.
    "n_vehiculos",
    "n_bus",
    "n_pesado_carga",
    "n_moto",
    "n_no_identificado",
    "n_interprovincial",
    "n_transporte_publico",
    "n_personas",
    "n_pasajeros",
    "n_peatones",
    "n_conductor_fugado",
    "edad_media_involucrados",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pre_endpoint_source(path: Path) -> pd.DataFrame:
    """Load only allowed raw fields through 2023 using predicate pushdown."""
    frame = pd.read_parquet(
        path,
        columns=EXPLANATION_RAW_COLUMNS,
        filters=[("FECHA", "<", pd.Timestamp("2024-01-01"))],
    )
    frame["FECHA"] = pd.to_datetime(frame["FECHA"], errors="raise")
    if frame.empty or frame["FECHA"].dt.year.max() > 2023:
        raise ValueError("Explainability source must contain only pre-2024 rows.")
    forbidden = {"target_multifatal", "FALLECIDOS", "LESIONADOS", "VEHICULOS_DANADOS"}
    if forbidden.intersection(frame.columns):
        raise ValueError("Outcome fields are forbidden in explainability inputs.")
    return frame


def partition_explainability_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the 2021-22 background pool and 2023 explanation pool."""
    year = pd.to_datetime(frame["FECHA"], errors="raise").dt.year
    background_pool = frame[year.isin([2021, 2022])].copy()
    validation_pool = frame[year == 2023].copy()
    if background_pool.empty or validation_pool.empty:
        raise ValueError("Explainability requires non-empty 2021-22 and 2023 partitions.")
    if set(pd.to_datetime(background_pool["FECHA"]).dt.year.unique()) - {2021, 2022}:
        raise ValueError("Background contains rows outside 2021-22.")
    if set(pd.to_datetime(validation_pool["FECHA"]).dt.year.unique()) != {2023}:
        raise ValueError("Explained examples must come exclusively from 2023 validation.")
    return background_pool, validation_pool


def deterministic_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("Sample size must be positive.")
    return frame.sample(n=min(size, len(frame)), replace=False, random_state=seed).sort_index()


def feature_group(feature: str) -> str:
    """Map the processed design matrix back to interpretable raw-variable groups."""
    if feature in {"mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos", "fin_de_semana"}:
        return "FECHA"
    if feature in {"hora_faltante", "hora_sin", "hora_cos", "nocturno"} or feature.startswith("franja_"):
        return "HORA"
    if feature.startswith("departamento_") or feature.startswith("region_"):
        return "DEPARTAMENTO / REGION_NATURAL"
    if feature == "via_freq" or feature.startswith("via_prefijo_"):
        return "CODIGO_VIA"
    if feature in {"LATITUD", "LONGITUD", "coord_faltante"}:
        return "COORDENADAS"
    if feature.startswith("clase_"):
        return "CLASE"
    if feature.startswith("zona_"):
        return "ZONA"
    if feature.startswith("red_vial_"):
        return "RED_VIAL"
    if feature.startswith("tipo_via_"):
        return "TIPO_VIA"
    if feature.startswith("clima_"):
        return "CLIMA"
    if feature.startswith("caracteristica_"):
        return "CARACTERISTICA_VIA"
    if feature.startswith("perfil_"):
        return "PERFIL_VIA"
    if feature.startswith("superficie_"):
        return "SUPERFICIE"
    if feature == "night_rural":
        return "HORA × ZONA"
    if feature == "rain_curve":
        return "CLIMA × CARACTERISTICA_VIA"
    if feature.startswith("road_type_zone_"):
        return "TIPO_VIA × ZONA"
    if feature.startswith("road_network_class_"):
        return "RED_VIAL × CLASE"
    if feature in {"n_vehiculos", "n_bus", "n_pesado_carga", "n_moto", "n_no_identificado", "n_interprovincial", "n_transporte_publico"}:
        return "VEHICULOS INVOLUCRADOS"
    if feature in {"n_personas", "n_pasajeros", "n_peatones", "n_conductor_fugado"}:
        return "PERSONAS INVOLUCRADAS"
    if feature in {"edad_media_involucrados", "edad_faltante"}:
        return "EDAD INVOLUCRADOS"
    raise ValueError(f"Processed feature has no interpretable group: {feature}")


def aggregate_global_explanations(
    shap_values: np.ndarray, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate signed SHAP contributions without losing group additivity."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix does not match the frozen feature contract.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP produced non-finite contributions.")

    feature_rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        group = feature_group(name)
        groups.setdefault(group, []).append(index)
        column = values[:, index]
        nonzero = np.abs(column) > 1e-15
        feature_rows.append(
            {
                "feature": name,
                "raw_variable_group": group,
                "mean_abs_shap": float(np.mean(np.abs(column))),
                "mean_signed_shap": float(np.mean(column)),
                "positive_contribution_share": float(np.mean(column[nonzero] > 0)) if nonzero.any() else 0.0,
            }
        )
    feature_table = pd.DataFrame(feature_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feature_table.insert(0, "rank", np.arange(1, len(feature_table) + 1))

    group_rows: list[dict[str, Any]] = []
    for group, indices in groups.items():
        # Sum within each example first: grouped SHAP values retain the signed
        # contribution of the raw variable and avoid double-counting one-hot
        # levels as independent variables.
        contribution = values[:, indices].sum(axis=1)
        signed = float(np.mean(contribution))
        nonzero = np.abs(contribution) > 1e-15
        group_rows.append(
            {
                "raw_variable_group": group,
                "processed_feature_count": len(indices),
                "mean_abs_grouped_shap": float(np.mean(np.abs(contribution))),
                "mean_signed_grouped_shap": signed,
                "positive_contribution_share": float(np.mean(contribution[nonzero] > 0)) if nonzero.any() else 0.0,
                "average_direction": "higher raw score" if signed > 0 else "lower raw score" if signed < 0 else "neutral",
            }
        )
    group_table = pd.DataFrame(group_rows).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
    total = float(group_table["mean_abs_grouped_shap"].sum())
    group_table["importance_share"] = group_table["mean_abs_grouped_shap"] / total if total > 0 else 0.0
    group_table.insert(0, "rank", np.arange(1, len(group_table) + 1))
    return group_table, feature_table


def _write_figure(group_table: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = group_table.sort_values("mean_abs_grouped_shap", ascending=True)
    colors = np.where(display["mean_signed_grouped_shap"] >= 0, "#C94F16", "#496B7C")
    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.92, bottom=0.12)
    axis.barh(display["raw_variable_group"], display["mean_abs_grouped_shap"], color=colors)
    axis.set_title("Importancia global · MLP definitiva", loc="left", weight="bold")
    axis.set_xlabel("Mean absolute grouped Gradient SHAP contribution", labelpad=10)
    axis.set_ylabel("")
    axis.grid(axis="x", color="#E1E5E2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.30,
        0.025,
        "Color: orange = positive mean signed contribution; blue = negative. Association, not causality.",
        fontsize=9,
        color="#5F6965",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def generate_final_explainability(
    root: Path = ROOT,
    *,
    background_size: int = BACKGROUND_SIZE,
    explanation_size: int = EXPLANATION_SIZE,
    shap_nsamples: int = SHAP_NSAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Generate canonical global evidence without retraining or endpoint access."""
    import joblib
    import shap
    import tensorflow as tf

    final_dir = root / "models" / "final"
    manifest_path = final_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("architecture", {}).get("weights_frozen") is not True:
        raise ValueError("Explainability requires the frozen canonical MLP.")
    for name, expected in manifest["artifact_hashes"].items():
        if sha256_file(final_dir / name) != expected:
            raise ValueError(f"Canonical artifact hash mismatch: {name}")

    source = load_pre_endpoint_source(root / manifest["dataset"]["path"])
    background_pool, validation_pool = partition_explainability_source(source)
    background_raw = deterministic_sample(background_pool, background_size, seed)
    validation_raw = deterministic_sample(validation_pool, explanation_size, seed + 1)

    scaler = joblib.load(final_dir / "scaler.joblib")
    encoders = joblib.load(final_dir / "encoders.joblib")
    feature_names = json.loads((final_dir / "feature_list.json").read_text(encoding="utf-8"))
    background = transform_features(background_raw, scaler, encoders)
    validation = transform_features(validation_raw, scaler, encoders)
    if background.columns.tolist() != feature_names or validation.columns.tolist() != feature_names:
        raise ValueError("Explainability matrices diverge from the frozen feature order.")

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model = tf.keras.models.load_model(final_dir / "model.keras")
    explainer = shap.GradientExplainer(model, background.to_numpy(dtype="float32"))
    shap_values = explainer.shap_values(
        validation.to_numpy(dtype="float32"), nsamples=shap_nsamples, rseed=seed
    )
    group_table, feature_table = aggregate_global_explanations(np.asarray(shap_values), feature_names)

    tables_dir = root / "report" / "tables"
    figures_dir = root / "report" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    group_path = tables_dir / "final_explainability_group_importance.csv"
    feature_path = tables_dir / "final_explainability_feature_importance.csv"
    provenance_path = tables_dir / "final_explainability_provenance.json"
    figure_path = figures_dir / "final_explainability_global.png"
    group_table.to_csv(group_path, index=False, float_format="%.10g")
    feature_table.to_csv(feature_path, index=False, float_format="%.10g")
    _write_figure(group_table, figure_path)

    index_digest = lambda values: hashlib.sha256(",".join(map(str, values)).encode("utf-8")).hexdigest()
    provenance: dict[str, Any] = {
        "schema_version": "1.0",
        "model_version": manifest["model_version"],
        "method": "Gradient SHAP (shap.GradientExplainer)",
        "shap_version": shap.__version__,
        "tensorflow_version": tf.__version__,
        "explained_output": "raw_mlp_sigmoid",
        "background_partition": "training_2021_2022_only",
        "explanation_partition": "validation_2023_only",
        "background_sample_size": int(len(background_raw)),
        "explanation_sample_size": int(len(validation_raw)),
        "shap_nsamples": int(shap_nsamples),
        "seed": int(seed),
        "sampling": "simple random sampling without replacement; sorted back to source index",
        "background_index_sha256": index_digest(background_raw.index.tolist()),
        "explanation_index_sha256": index_digest(validation_raw.index.tolist()),
        "source_projection": EXPLANATION_RAW_COLUMNS,
        "source_predicate": "FECHA < 2024-01-01 (Parquet predicate pushdown)",
        "labels_loaded": False,
        "endpoint_2024_2025_data_loaded": False,
        "model_selection_performed": False,
        "weights_or_calibration_modified": False,
        "grouping_rule": "Sum signed processed-feature SHAP values within each raw-variable group per example, then aggregate absolute and signed means.",
        "direction_interpretation": "Mean signed contribution over sampled 2023 validation examples relative to the 2021-22 background; it is not a monotonic effect.",
        "association_not_causality": True,
        "local_explanations_generated": False,
        "limitations": "Global associations describe this frozen model and sample only. They do not identify causes, interventions, or feature effects for an individual case.",
        "generator_sha256": sha256_file(Path(__file__)),
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_paths = [group_path, feature_path, provenance_path, figure_path]
    explainability_hashes = {
        str(path.relative_to(root)): sha256_file(path) for path in artifact_paths
    }
    manifest["explainability"] = {
        **{key: provenance[key] for key in (
            "method", "explained_output", "background_partition", "explanation_partition",
            "background_sample_size", "explanation_sample_size", "shap_nsamples", "seed",
            "association_not_causality", "local_explanations_generated", "generator_sha256",
        )},
        "status": "generated",
    }
    manifest["explainability_artifact_hashes"] = explainability_hashes
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"provenance": provenance, "artifact_hashes": explainability_hashes}


if __name__ == "__main__":
    result = generate_final_explainability()
    print(json.dumps(result, ensure_ascii=False, indent=2))
