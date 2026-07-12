"""Read-only inference and observational context for the Streamlit application.

The application consumes the hash-verified canonical bundle.  It never builds
features, cleans data, calibrates a model, or writes artifacts at runtime.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import assign_time_band, normalize_clase, normalize_clima, normalize_zona, parse_hour_value
from src.final_model_bundle import CanonicalModelBundle, sha256_file


FINAL_MODEL_DIR = ROOT / "models" / "final"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
BASE_PATH = PROCESSED_DIR / "base_limpia.parquet"
DEMO_PATH = PROCESSED_DIR / "demo_cases.csv"
GEOJSON_PATH = ROOT / "data" / "geo" / "peru_departamentos_simple.geojson"
INFERENCE_DATE_MIN = pd.Timestamp("2021-01-01")
INFERENCE_DATE_MAX = pd.Timestamp("2025-12-31")
MINIMUM_COMPARISON_SUPPORT = 30
ROAD_CODE_PATTERN = re.compile(r"^(?:PE|[A-Z]{2})-[A-Z0-9]{1,8}(?:-[A-Z0-9]{1,8}){0,2}$")


class RuntimeArtifactError(RuntimeError):
    """A frozen application artifact is missing, corrupt, or incompatible."""


class InputContractError(ValueError):
    """A submitted record does not satisfy the canonical raw-input contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeArtifactError(
            f"No se encontró {label} en {path}. Restaurá el artefacto congelado antes de iniciar la app."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(
            f"No se pudo leer {label} en {path}. Verificá que el archivo sea JSON válido y no esté dañado."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeArtifactError(f"{label} no contiene un objeto JSON válido: {path}")
    return payload


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    manifest = _read_json(FINAL_MODEL_DIR / "manifest.json", "el manifiesto canónico")
    if manifest.get("model_version") != "canonical-1.0.0":
        raise RuntimeArtifactError(
            "La versión del bundle no es compatible con esta interfaz. Regenerá el bundle fuera de la app."
        )
    return manifest


@lru_cache(maxsize=1)
def load_feature_schema() -> dict[str, Any]:
    schema = _read_json(FINAL_MODEL_DIR / "feature_schema.json", "el esquema de entrada")
    manifest = load_manifest()
    if int(schema.get("processed_feature_count", -1)) != int(manifest.get("feature_count", -2)):
        raise RuntimeArtifactError(
            "El esquema y el manifiesto discrepan en el número de features. No se ejecutó inferencia."
        )
    return schema


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    thresholds = _read_json(FINAL_MODEL_DIR / "thresholds.json", "los umbrales canónicos")
    for scale in ("raw", "calibrated"):
        try:
            value = float(thresholds[scale]["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeArtifactError(f"Falta el umbral {scale} en el bundle canónico.") from exc
        if not 0.0 <= value <= 1.0:
            raise RuntimeArtifactError(f"El umbral {scale} está fuera del intervalo [0, 1].")
    return thresholds


@lru_cache(maxsize=1)
def load_prediction_stack() -> CanonicalModelBundle:
    """Load and hash-verify the canonical runtime without mutating disk."""
    try:
        runtime = CanonicalModelBundle(FINAL_MODEL_DIR, verify_hashes=True)
    except FileNotFoundError as exc:
        raise RuntimeArtifactError(
            "Falta un archivo de models/final/. Restaurá el bundle canónico completo; la app no lo reconstruye."
        ) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeArtifactError(
            f"El bundle canónico no superó la verificación de integridad: {exc}"
        ) from exc
    manifest = load_manifest()
    schema = load_feature_schema()
    if int(runtime.model.input_shape[-1]) != int(manifest["feature_count"]):
        raise RuntimeArtifactError("El ancho de entrada del modelo no coincide con el manifiesto.")
    if runtime.encoders.get("feature_list") != schema.get("processed_feature_order"):
        raise RuntimeArtifactError("El orden de features del encoder no coincide con el esquema persistido.")
    return runtime


def load_threshold() -> float:
    """Return the user-facing calibrated threshold (never the raw threshold)."""
    return float(load_thresholds()["calibrated"]["value"])


@lru_cache(maxsize=1)
def load_explainability_artifacts() -> dict[str, Any]:
    """Load hash-verified global evidence; never compute explanations at runtime."""
    required = {
        "groups": TABLES_DIR / "final_explainability_group_importance.csv",
        "features": TABLES_DIR / "final_explainability_feature_importance.csv",
        "provenance": TABLES_DIR / "final_explainability_provenance.json",
        "figure": FIGURES_DIR / "final_explainability_global.png",
    }
    expected_hashes = load_manifest().get("explainability_artifact_hashes", {})
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeArtifactError("El manifiesto no referencia la explicabilidad canónica.")
    for path in required.values():
        relative = str(path.relative_to(ROOT))
        expected = expected_hashes.get(relative)
        if not path.exists():
            raise RuntimeArtifactError(f"Falta la evidencia de explicabilidad {relative}.")
        if not expected or sha256_file(path) != expected:
            raise RuntimeArtifactError(f"La evidencia de explicabilidad {relative} no coincide con el manifiesto.")
    try:
        groups = pd.read_csv(required["groups"])
        features = pd.read_csv(required["features"])
        provenance = _read_json(required["provenance"], "la procedencia de explicabilidad")
    except (OSError, ValueError) as exc:
        raise RuntimeArtifactError(f"No se pudo leer la explicabilidad canónica: {exc}") from exc
    required_group_columns = {
        "rank", "raw_variable_group", "processed_feature_count", "mean_abs_grouped_shap",
        "mean_signed_grouped_shap", "positive_contribution_share", "average_direction", "importance_share",
    }
    if not required_group_columns.issubset(groups.columns):
        raise RuntimeArtifactError("La tabla de explicabilidad agrupada no cumple su esquema.")
    numeric = groups[["mean_abs_grouped_shap", "mean_signed_grouped_shap", "importance_share"]]
    if groups.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeArtifactError("La explicabilidad agrupada contiene valores vacíos o no finitos.")
    if provenance.get("explanation_partition") != "validation_2023_only":
        raise RuntimeArtifactError("La explicabilidad no respeta la partición de validación 2023.")
    if provenance.get("endpoint_2024_2025_data_loaded") is not False or provenance.get("labels_loaded") is not False:
        raise RuntimeArtifactError("La explicabilidad declara acceso a datos o etiquetas no permitidos.")
    return {"groups": groups, "features": features, "provenance": provenance, "figure": required["figure"]}


def load_input_options() -> dict[str, list[str | None]]:
    """Expose raw-form representatives derived from persisted categories.

    Some encoders store normalized labels that are not themselves accepted raw
    aliases (for example ``RECTO`` derives from ``TRAMO RECTO``).  Map those
    labels back to one valid raw representative so the UI cannot silently send
    a normalized label into the wrong bucket.
    """
    encoders = load_prediction_stack().encoders
    mapping = {
        "DEPARTAMENTO": "departamento_categories",
        "CLASE": "clase_categories",
        "ZONA": "zona_categories",
        "RED_VIAL": "red_vial_categories",
        "TIPO_VIA": "tipo_via_categories",
        "CLIMA": "clima_categories",
        "CARACTERISTICA_VIA": "caracteristica_categories",
        "PERFIL_VIA": "perfil_categories",
        "SUPERFICIE": "superficie_categories",
    }
    raw_representatives: dict[str, dict[str, str | None]] = {
        "CLASE": {"DESCONOCIDO": None},
        "TIPO_VIA": {"EXPRESA": "VIA EXPRESA", "DESCONOCIDO": None},
        "CARACTERISTICA_VIA": {
            "RECTO": "TRAMO RECTO",
            "INTERSECCION": "INTERSECCIÓN",
            "ESTRUCTURA": "PUENTE",
            "DESCONOCIDO": None,
        },
        "SUPERFICIE": {"PAVIMENTADA": "ASFALTADA", "DESCONOCIDO": None},
        "ZONA": {"DESCONOCIDO": None},
        "RED_VIAL": {"DESCONOCIDO": None},
        "CLIMA": {"DESCONOCIDO": None},
        "PERFIL_VIA": {"DESCONOCIDO": None},
    }
    schema = load_feature_schema()
    nullable = {str(field["name"]): bool(field["nullable"]) for field in schema["required_raw_fields"]}
    options: dict[str, list[str | None]] = {}
    for field, key in mapping.items():
        values = encoders.get(key)
        if not isinstance(values, list) or not values:
            raise RuntimeArtifactError(f"El encoder canónico no declara categorías para {field}.")
        representatives = [raw_representatives.get(field, {}).get(str(value), str(value)) for value in values]
        options[field] = [value for value in representatives if value is not None or nullable.get(field, False)]
    return options


def load_known_road_codes() -> list[str]:
    """Return searchable road codes persisted with the training encoder."""
    frequency_map = load_prediction_stack().encoders.get("via_frequency_map")
    if not isinstance(frequency_map, dict) or not frequency_map:
        raise RuntimeArtifactError("El encoder canónico no declara códigos de vía conocidos.")
    return sorted(str(value) for value in frequency_map if str(value) != "DESCONOCIDO")


def normalize_road_code(value: Any) -> str:
    """Normalize a road code while preserving the canonical unknown bucket."""
    if value is None or pd.isna(value) or not str(value).strip():
        return "DESCONOCIDO"
    return str(value).strip().upper()


def is_known_road_code(value: Any) -> bool:
    normalized = normalize_road_code(value)
    return normalized == "DESCONOCIDO" or normalized in set(load_known_road_codes())


def _point_on_segment(
    longitude: float,
    latitude: float,
    start: list[float] | tuple[float, float],
    end: list[float] | tuple[float, float],
    tolerance: float = 1e-9,
) -> bool:
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    cross = (longitude - x1) * (y2 - y1) - (latitude - y1) * (x2 - x1)
    if abs(cross) > tolerance * max(1.0, abs(x2 - x1), abs(y2 - y1)):
        return False
    return (
        min(x1, x2) - tolerance <= longitude <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= latitude <= max(y1, y2) + tolerance
    )


def point_in_ring(longitude: float, latitude: float, ring: list[list[float]]) -> int:
    """Return 0 outside, 1 inside, or 2 on the boundary of a linear ring."""
    if len(ring) < 4:
        return 0
    inside = False
    previous = ring[-1]
    for current in ring:
        if _point_on_segment(longitude, latitude, previous, current):
            return 2
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = (y1 > latitude) != (y2 > latitude)
        if crosses:
            intersection = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < intersection:
                inside = not inside
        previous = current
    return 1 if inside else 0


def point_in_polygon(longitude: float, latitude: float, polygon: list[list[list[float]]]) -> bool:
    """Test a point against a GeoJSON polygon, accepting polygon boundaries."""
    if not polygon:
        return False
    outer_state = point_in_ring(longitude, latitude, polygon[0])
    if outer_state == 0:
        return False
    if outer_state == 2:
        return True
    for hole in polygon[1:]:
        hole_state = point_in_ring(longitude, latitude, hole)
        if hole_state == 2:
            return True
        if hole_state == 1:
            return False
    return True


def point_in_geometry(longitude: float, latitude: float, geometry: dict[str, Any]) -> bool:
    """Test a point against a GeoJSON Polygon or MultiPolygon."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        return point_in_polygon(longitude, latitude, coordinates)
    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        return any(point_in_polygon(longitude, latitude, polygon) for polygon in coordinates)
    return False


@lru_cache(maxsize=1)
def load_department_geometries() -> dict[str, dict[str, Any]]:
    """Load the versioned Peru departmental polygons used by input validation."""
    try:
        payload = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeArtifactError(
            f"No se encontró el mapa departamental en {GEOJSON_PATH}. Restaurá el GeoJSON canónico."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(f"No se pudo leer el mapa departamental: {exc}") from exc
    geometries: dict[str, dict[str, Any]] = {}
    for feature in payload.get("features", []):
        department = str(feature.get("properties", {}).get("NOMBDEP", "")).strip().upper()
        geometry = feature.get("geometry")
        if department and isinstance(geometry, dict):
            geometries[department] = geometry
    if not geometries:
        raise RuntimeArtifactError("El mapa departamental no contiene geometrías utilizables.")
    return geometries


def departments_for_point(latitude: float, longitude: float) -> list[str]:
    """Return every department covering a point; borders may match two regions."""
    return sorted(
        department
        for department, geometry in load_department_geometries().items()
        if point_in_geometry(float(longitude), float(latitude), geometry)
    )


def validate_peru_location(latitude: float, longitude: float, department: str) -> None:
    """Validate country coverage and department/coordinate coherence."""
    matches = departments_for_point(latitude, longitude)
    if not matches:
        raise InputContractError(
            "Las coordenadas están fuera del territorio continental e insular representado en el mapa del Perú."
        )
    selected = str(department).strip().upper()
    if selected not in matches:
        raise InputContractError(
            "Las coordenadas no corresponden al departamento seleccionado. "
            f"El mapa las ubica en {', '.join(matches)}; revisá departamento, latitud y longitud."
        )


def _validate_records(records: pd.DataFrame) -> pd.DataFrame:
    schema = load_feature_schema()
    required = [str(field["name"]) for field in schema["required_raw_fields"]]
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise InputContractError(f"Faltan campos requeridos: {', '.join(missing)}.")
    if records.empty:
        raise InputContractError("No se recibió ningún registro para estimar.")

    clean = records.copy()
    clean["FECHA"] = pd.to_datetime(clean["FECHA"], errors="coerce")
    if clean["FECHA"].isna().any():
        raise InputContractError("La fecha no es válida.")
    out_of_window = ~clean["FECHA"].between(INFERENCE_DATE_MIN, INFERENCE_DATE_MAX)
    if out_of_window.any():
        raise InputContractError(
            "La fecha debe estar entre el 01/01/2021 y el 31/12/2025, periodo cubierto por la evidencia académica."
        )
    parsed_hours = clean["HORA"].apply(parse_hour_value)
    if parsed_hours.isna().any():
        raise InputContractError("La hora debe estar entre 00:00 y 23:59.")

    latitudes = pd.to_numeric(clean["LATITUD"], errors="coerce")
    longitudes = pd.to_numeric(clean["LONGITUD"], errors="coerce")
    mismatched_coordinates = latitudes.isna() ^ longitudes.isna()
    if mismatched_coordinates.any():
        raise InputContractError("Ingresá ambas coordenadas; no se acepta solo una.")
    if (latitudes.isna() | longitudes.isna()).any():
        raise InputContractError(
            "La latitud y la longitud son obligatorias para el modelo canónico; ingresá ambas coordenadas."
        )
    if ((latitudes.dropna() < -90) | (latitudes.dropna() > 90)).any():
        raise InputContractError("La latitud debe estar entre -90 y 90.")
    if ((longitudes.dropna() < -180) | (longitudes.dropna() > 180)).any():
        raise InputContractError("La longitud debe estar entre -180 y 180.")
    clean["LATITUD"] = latitudes
    clean["LONGITUD"] = longitudes

    nullable = {str(field["name"]): bool(field["nullable"]) for field in schema["required_raw_fields"]}
    categorical = [
        "DEPARTAMENTO", "CLASE", "ZONA", "RED_VIAL", "TIPO_VIA", "CLIMA",
        "CARACTERISTICA_VIA", "PERFIL_VIA", "SUPERFICIE",
    ]
    for column in categorical:
        empty = clean[column].notna() & clean[column].astype(str).str.strip().eq("")
        if empty.any() or (clean[column].isna().any() and not nullable.get(column, False)):
            raise InputContractError(f"El campo {column} no puede quedar vacío.")
    clean["CODIGO_VIA"] = clean["CODIGO_VIA"].map(normalize_road_code)
    invalid_road_code = ~clean["CODIGO_VIA"].map(
        lambda value: value == "DESCONOCIDO" or bool(ROAD_CODE_PATTERN.fullmatch(value))
    )
    if invalid_road_code.any():
        raise InputContractError(
            "El código de vía debe usar un formato como PE-1N, PE-3S o AM-103; "
            "dejalo como NO INFORMADO si no está disponible."
        )
    for row in clean.itertuples(index=False):
        validate_peru_location(float(row.LATITUD), float(row.LONGITUD), str(row.DEPARTAMENTO))
    return clean


def predict_records(records: pd.DataFrame) -> pd.DataFrame:
    """Predict from the frozen bundle using distinct calibrated/raw contracts."""
    clean = _validate_records(records)
    runtime = load_prediction_stack()
    try:
        predictions = runtime.predict_dataframe(clean)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeArtifactError(
            f"El modelo no pudo procesar el registro con el contrato canónico: {exc}"
        ) from exc
    thresholds = load_thresholds()
    output = records.copy()
    output["calibrated_probability"] = predictions["calibrated_probability"].to_numpy()
    output["calibrated_threshold"] = float(thresholds["calibrated"]["value"])
    output["priority_decision"] = np.where(
        predictions["calibrated_prediction"].to_numpy() == 1,
        "PRIORIZAR REVISIÓN",
        "PRIORIDAD ESTÁNDAR",
    )
    output["raw_probability"] = predictions["raw_probability"].to_numpy()
    output["raw_threshold"] = float(thresholds["raw"]["value"])
    output["raw_prediction"] = predictions["raw_prediction"].to_numpy()
    output["calibration_method"] = runtime.calibration_method
    return output


def _verified_dataset_path() -> Path:
    manifest = load_manifest()
    path = ROOT / str(manifest["dataset"]["path"])
    if not path.exists():
        raise RuntimeArtifactError(
            f"No se encontró el dataset limpio en {path}. La app no ejecuta limpieza automática."
        )
    actual_hash = sha256_file(path)
    if actual_hash != manifest["dataset"]["sha256"]:
        raise RuntimeArtifactError(
            "El dataset limpio no coincide con el hash del modelo. Restaurá la versión canónica antes de usar la app."
        )
    return path


@lru_cache(maxsize=1)
def load_clean_dataset() -> pd.DataFrame:
    try:
        frame = pd.read_parquet(_verified_dataset_path())
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeArtifactError(
            f"No se pudo abrir el dataset limpio. Verificá pyarrow y el archivo parquet: {exc}"
        ) from exc
    required = {"FECHA", "HORA", "DEPARTAMENTO", "target_multifatal"}
    if not required.issubset(frame.columns):
        raise RuntimeArtifactError("El dataset limpio no contiene las columnas mínimas de la interfaz.")
    return frame


@lru_cache(maxsize=1)
def load_demo_cases() -> pd.DataFrame:
    if not DEMO_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DEMO_PATH)
    except (OSError, ValueError) as exc:
        raise RuntimeArtifactError(f"No se pudieron leer los casos ilustrativos: {exc}") from exc


def wilson_interval(positives: int, support: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if support <= 0:
        return None, None
    proportion = positives / support
    denominator = 1 + (z * z / support)
    centre = (proportion + (z * z / (2 * support))) / denominator
    margin = z * np.sqrt((proportion * (1 - proportion) / support) + (z * z / (4 * support * support))) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _observed_row(label: str, subset: pd.DataFrame, dimensions: str) -> dict[str, Any]:
    support = int(len(subset))
    positives = int(subset["target_multifatal"].sum()) if support else 0
    lower, upper = wilson_interval(positives, support)
    return {
        "comparador": label,
        "tasa_multifatal": (positives / support) if support else None,
        "soporte": support,
        "positivos": positives,
        "ci_95_inf": lower,
        "ci_95_sup": upper,
        "dimensiones": dimensions,
        "fuente": "ONSV 2021–2025 (preliminar)",
    }


def historical_comparison(record: pd.Series | dict[str, Any], predicted_probability: float) -> pd.DataFrame:
    """Return observational context, not a causal explanation or individual truth."""
    df = load_clean_dataset().copy()
    df["clase_norm"] = df["CLASE"].apply(normalize_clase)
    df["zona_norm"] = df["ZONA"].apply(normalize_zona)
    df["clima_norm"] = df["CLIMA"].apply(normalize_clima)

    data = dict(record)
    clase = normalize_clase(data.get("CLASE"))
    zona = normalize_zona(data.get("ZONA"))
    clima = normalize_clima(data.get("CLIMA"))
    departamento = str(data.get("DEPARTAMENTO", "")).strip().upper()
    hour = parse_hour_value(data.get("HORA"))
    franja = assign_time_band(hour)

    exact_mask = (
        (df["clase_norm"] == clase)
        & (df["zona_norm"] == zona)
        & (df["DEPARTAMENTO"] == departamento)
    )
    if pd.notna(hour):
        bands = {
            "MADRUGADA": (0, 5), "MANANA": (6, 11), "TARDE": (12, 17), "NOCHE": (18, 23),
        }
        if franja in bands:
            start, end = bands[franja]
            exact_mask &= df["hora_entera"].between(start, end)

    rows = [
        {
            "comparador": "Probabilidad calibrada del modelo",
            "tasa_multifatal": float(predicted_probability),
            "soporte": None,
            "positivos": None,
            "ci_95_inf": None,
            "ci_95_sup": None,
            "dimensiones": "Registro ingresado",
            "fuente": "MLP canónica + calibración Platt",
        },
        _observed_row("Referencia global", df, "Todos los siniestros fatales"),
        _observed_row("Misma clase", df.loc[df["clase_norm"] == clase], f"CLASE={clase}"),
        _observed_row("Misma zona", df.loc[df["zona_norm"] == zona], f"ZONA={zona}"),
        _observed_row("Mismo clima", df.loc[df["clima_norm"] == clima], f"CLIMA={clima}"),
        _observed_row(
            "Subgrupo coincidente",
            df.loc[exact_mask],
            f"CLASE={clase}; ZONA={zona}; DEPARTAMENTO={departamento}; FRANJA={franja}",
        ),
    ]
    comparison = pd.DataFrame(rows)
    comparison = mask_unsupported_historical_rates(comparison, MINIMUM_COMPARISON_SUPPORT)
    for column in ("tasa_multifatal", "ci_95_inf", "ci_95_sup"):
        comparison[f"{column}_pct"] = pd.to_numeric(comparison[column], errors="coerce") * 100
    return comparison


def mask_unsupported_historical_rates(
    comparison: pd.DataFrame, minimum_support: int = MINIMUM_COMPARISON_SUPPORT
) -> pd.DataFrame:
    """Mask the matching subgroup when its observed support is below the declared minimum."""
    required = {"comparador", "soporte", "tasa_multifatal", "ci_95_inf", "ci_95_sup"}
    if not required.issubset(comparison.columns):
        missing = sorted(required - set(comparison.columns))
        raise ValueError(f"Historical comparison is missing columns: {', '.join(missing)}")
    masked = comparison.copy()
    subgroup = masked["comparador"].eq("Subgrupo coincidente")
    supported = pd.to_numeric(masked["soporte"], errors="coerce").ge(minimum_support)
    masked["soporte_suficiente"] = True
    masked.loc[subgroup, "soporte_suficiente"] = supported.loc[subgroup]
    masked.loc[subgroup & ~supported, ["tasa_multifatal", "ci_95_inf", "ci_95_sup"]] = np.nan
    return masked


def regional_summary(minimum_support: int = 30) -> pd.DataFrame:
    df = load_clean_dataset()
    rows = []
    for department, subset in df.groupby("DEPARTAMENTO", dropna=False):
        support = int(len(subset))
        positives = int(subset["target_multifatal"].sum())
        lower, upper = wilson_interval(positives, support)
        rows.append(
            {
                "DEPARTAMENTO": str(department),
                "siniestros_fatales": support,
                "multifatales": positives,
                "tasa_multifatal": positives / support,
                "ci_95_inf": lower,
                "ci_95_sup": upper,
                "soporte_suficiente": support >= minimum_support,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["soporte_suficiente", "tasa_multifatal", "siniestros_fatales"], ascending=[False, False, False]
    )


def mask_unsupported_regional_rates(
    regional: pd.DataFrame, minimum_support: int = 30
) -> pd.DataFrame:
    """Hide unstable regional rate estimates from tables and downloads."""
    required = {
        "siniestros_fatales", "tasa_multifatal", "ci_95_inf", "ci_95_sup",
    }
    if not required.issubset(regional.columns):
        missing = sorted(required - set(regional.columns))
        raise ValueError(f"Regional table is missing columns: {', '.join(missing)}")
    masked = regional.copy()
    supported = masked["siniestros_fatales"].astype(int) >= minimum_support
    masked["soporte_suficiente"] = supported
    masked.loc[~supported, ["tasa_multifatal", "ci_95_inf", "ci_95_sup"]] = np.nan
    return masked
