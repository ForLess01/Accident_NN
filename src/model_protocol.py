"""Leakage-safe feature contract and chronological evaluation protocol.

This module contains no model fitting, so partition boundaries and the feature
contract can be verified without TensorFlow.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import (
    CARACTERISTICA_CATEGORIES,
    CLASE_CATEGORIES,
    CLIMA_CATEGORIES,
    FRANJA_CATEGORIES,
    PERFIL_CATEGORIES,
    RED_VIAL_CATEGORIES,
    REGION_CATEGORIES,
    SUPERFICIE_CATEGORIES,
    TIPO_VIA_CATEGORIES,
    ZONA_CATEGORIES,
    REGION_NATURAL_BY_DEPARTAMENTO,
    add_one_hot,
    assign_time_band,
    extract_via_prefix,
    normalize_caracteristica,
    normalize_clase,
    normalize_clima,
    normalize_perfil,
    normalize_red_vial,
    normalize_superficie,
    normalize_text_value,
    normalize_tipo_via,
    normalize_zona,
    parse_hour_value,
)


EXCLUDED_COLUMNS = frozenset(
    {
        "target_multifatal",
        "FALLECIDOS",
        "LESIONADOS",
        "VEHICULOS_DANADOS",
        "CAUSA_FACTOR",
        "CAUSA_ESPECIFICA",
        "SENAL_VERTICAL",
        "SENAL_HORIZONTAL",
        "CODIGO_SINIESTRO",
        "PROVINCIA",
        "DISTRITO",
        "ZONIFICACION",
    }
)
VEHICLE_COUNT_COLUMNS = [
    "n_vehiculos",
    "n_bus",
    "n_pesado_carga",
    "n_moto",
    "n_no_identificado",
    "n_interprovincial",
    "n_transporte_publico",
]
COMPANION_COUNT_COLUMNS = VEHICLE_COUNT_COLUMNS
PERSONAS_DERIVED_COLUMNS = frozenset(
    {"n_personas", "n_pasajeros", "n_peatones", "n_conductor_fugado", "edad_media_involucrados", "edad_faltante"}
)
CONTINUOUS_COLUMNS = ["LATITUD", "LONGITUD", "via_freq", *VEHICLE_COUNT_COLUMNS]


def split_chronological(base: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Create the predeclared 2021--22 / 2023 / 2024--25 chronological partitions."""
    frame = base.copy()
    frame["FECHA"] = pd.to_datetime(frame["FECHA"], errors="coerce")
    if frame["FECHA"].isna().any():
        raise ValueError("Chronological protocol requires a valid FECHA for every row.")
    if "target_multifatal" not in frame:
        raise ValueError("Chronological protocol requires target_multifatal.")

    year = frame["FECHA"].dt.year
    train = frame[year.isin([2021, 2022])].copy()
    validation = frame[year == 2023].copy()
    test = frame[year.isin([2024, 2025])].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("The protocol needs non-empty 2021--22, 2023 and 2024--25 partitions.")

    def xy(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return (
            partition.drop(columns=[column for column in EXCLUDED_COLUMNS if column in partition]),
            partition["target_multifatal"].astype("int8"),
        )

    X_train, y_train = xy(train)
    X_validation, y_validation = xy(validation)
    X_test, y_test = xy(test)
    return {
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_validation_raw": X_validation,
        "y_validation": y_validation,
        "X_test_raw": X_test,
        "y_test": y_test,
    }


def _one_hot_combinations(base: pd.DataFrame, column: str, categories: list[str], prefix: str) -> pd.DataFrame:
    return add_one_hot(base, column, categories, prefix)


def derive_base_features(df: pd.DataFrame, via_frequency_map: dict[str, float] | None = None) -> pd.DataFrame:
    """Derive only pre-impact or initial-scene features in the final contract."""
    features = pd.DataFrame(index=df.index)
    fecha = pd.to_datetime(df["FECHA"], errors="coerce")
    month = fecha.dt.month.astype("float64")
    weekday = fecha.dt.dayofweek.astype("float64")
    # Periodic encodings prevent December/January and Sunday/Monday from being
    # represented as far apart; year is deliberately excluded in a future test.
    features["mes_sin"] = np.where(month.isna(), 0.0, np.sin(2 * np.pi * month / 12))
    features["mes_cos"] = np.where(month.isna(), 0.0, np.cos(2 * np.pi * month / 12))
    features["dia_semana_sin"] = np.where(weekday.isna(), 0.0, np.sin(2 * np.pi * weekday / 7))
    features["dia_semana_cos"] = np.where(weekday.isna(), 0.0, np.cos(2 * np.pi * weekday / 7))
    features["fin_de_semana"] = weekday.isin([5, 6]).astype("int8")

    hour = df["hora_entera"] if "hora_entera" in df else df["HORA"].apply(parse_hour_value)
    hour = pd.to_numeric(hour, errors="coerce")
    hour_filled = hour.fillna(0)
    features["hora_faltante"] = hour.isna().astype("int8")
    features["hora_sin"] = np.where(hour.isna(), 0.0, np.sin(2 * np.pi * hour_filled / 24))
    features["hora_cos"] = np.where(hour.isna(), 0.0, np.cos(2 * np.pi * hour_filled / 24))
    features["nocturno"] = ((hour >= 20) | (hour <= 5)).fillna(False).astype("int8")
    features["franja"] = hour.apply(assign_time_band)

    departamento = df["DEPARTAMENTO"].map(normalize_text_value).fillna("DESCONOCIDO")
    features["DEPARTAMENTO"] = departamento
    features["region_natural"] = departamento.map(REGION_NATURAL_BY_DEPARTAMENTO).fillna("DESCONOCIDO")
    codigo_via = df["CODIGO_VIA"].map(normalize_text_value).fillna("DESCONOCIDO")
    features["CODIGO_VIA"] = codigo_via
    features["via_prefijo"] = codigo_via.apply(extract_via_prefix)
    features["via_freq"] = 0.0 if via_frequency_map is None else codigo_via.map(via_frequency_map).fillna(0.0)

    latitud = pd.to_numeric(df["LATITUD"], errors="coerce")
    longitud = pd.to_numeric(df["LONGITUD"], errors="coerce")
    features["coord_faltante"] = (latitud.isna() | longitud.isna()).astype("int8")
    features["LATITUD"] = latitud
    features["LONGITUD"] = longitud
    features["CLASE"] = df["CLASE"].apply(normalize_clase)
    features["ZONA"] = df["ZONA"].apply(normalize_zona)
    features["RED_VIAL"] = df["RED_VIAL"].apply(normalize_red_vial)
    features["TIPO_VIA"] = df["TIPO_VIA"].apply(normalize_tipo_via)
    features["CLIMA"] = df["CLIMA"].apply(normalize_clima)
    features["CARACTERISTICA_VIA"] = df["CARACTERISTICA_VIA"].apply(normalize_caracteristica)
    features["PERFIL_VIA"] = df["PERFIL_VIA"].apply(normalize_perfil)
    features["SUPERFICIE"] = df["SUPERFICIE"].apply(normalize_superficie)

    features["night_rural"] = ((features["nocturno"] == 1) & (features["ZONA"] == "RURAL")).astype("int8")
    features["rain_curve"] = ((features["CLIMA"] == "LLUVIOSO") & (features["CARACTERISTICA_VIA"] == "CURVA")).astype("int8")
    features["road_type_zone"] = features["TIPO_VIA"] + "__" + features["ZONA"]
    features["road_network_class"] = features["RED_VIAL"] + "__" + features["CLASE"]

    # VEHICULOS aggregates are consolidated retrospective facts. PERSONAS is
    # structurally prohibited because its cardinality reconstructs the target.
    for column in VEHICLE_COUNT_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce") if column in df else pd.Series(np.nan, index=df.index)
        features[column] = values.fillna(0).clip(lower=0).astype("float64")
    return features


def fit_preprocessor(train_df: pd.DataFrame) -> tuple[StandardScaler, dict[str, Any]]:
    preliminary = derive_base_features(train_df)
    lat_median = float(preliminary["LATITUD"].median())
    lon_median = float(preliminary["LONGITUD"].median())
    via_frequency_map = preliminary["CODIGO_VIA"].value_counts(normalize=True, dropna=False).astype(float).to_dict()
    base = derive_base_features(train_df, via_frequency_map)
    base["LATITUD"] = base["LATITUD"].fillna(lat_median)
    base["LONGITUD"] = base["LONGITUD"].fillna(lon_median)
    scaler = StandardScaler().fit(base[CONTINUOUS_COLUMNS])
    encoders: dict[str, Any] = {
        "lat_median": lat_median,
        "lon_median": lon_median,
        "via_frequency_map": via_frequency_map,
        "departamento_categories": sorted(base["DEPARTAMENTO"].dropna().unique().tolist()),
        "via_prefijo_categories": sorted(base["via_prefijo"].dropna().unique().tolist()),
        "region_categories": REGION_CATEGORIES,
        "franja_categories": FRANJA_CATEGORIES,
        "clase_categories": CLASE_CATEGORIES,
        "zona_categories": ZONA_CATEGORIES,
        "red_vial_categories": RED_VIAL_CATEGORIES,
        "tipo_via_categories": TIPO_VIA_CATEGORIES,
        "clima_categories": CLIMA_CATEGORIES,
        "caracteristica_categories": CARACTERISTICA_CATEGORIES,
        "perfil_categories": PERFIL_CATEGORIES,
        "superficie_categories": SUPERFICIE_CATEGORIES,
        "road_type_zone_categories": [f"{road}__{zone}" for road in TIPO_VIA_CATEGORIES for zone in ZONA_CATEGORIES],
        "road_network_class_categories": [f"{network}__{kind}" for network in RED_VIAL_CATEGORIES for kind in CLASE_CATEGORIES],
        "continuous_columns": CONTINUOUS_COLUMNS,
    }
    encoders["feature_list"] = transform_features(train_df, scaler, encoders).columns.tolist()
    return scaler, encoders


def transform_features(df: pd.DataFrame, scaler: StandardScaler, encoders: dict[str, Any]) -> pd.DataFrame:
    base = derive_base_features(df, encoders["via_frequency_map"])
    base["LATITUD"] = base["LATITUD"].fillna(float(encoders["lat_median"]))
    base["LONGITUD"] = base["LONGITUD"].fillna(float(encoders["lon_median"]))
    output = pd.DataFrame(index=base.index)
    scaled = scaler.transform(base[encoders["continuous_columns"]])
    for idx, column in enumerate(encoders["continuous_columns"]):
        output[column] = scaled[:, idx]
    for column in ["mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos", "fin_de_semana", "hora_faltante", "hora_sin", "hora_cos", "nocturno", "coord_faltante", "night_rural", "rain_curve"]:
        output[column] = base[column].fillna(0)
    one_hot_parts = [
        add_one_hot(base, "DEPARTAMENTO", encoders["departamento_categories"], "departamento"),
        add_one_hot(base, "region_natural", encoders["region_categories"], "region"),
        add_one_hot(base, "via_prefijo", encoders["via_prefijo_categories"], "via_prefijo"),
        add_one_hot(base, "franja", encoders["franja_categories"], "franja"),
        add_one_hot(base, "CLASE", encoders["clase_categories"], "clase"),
        add_one_hot(base, "ZONA", encoders["zona_categories"], "zona"),
        add_one_hot(base, "RED_VIAL", encoders["red_vial_categories"], "red_vial"),
        add_one_hot(base, "TIPO_VIA", encoders["tipo_via_categories"], "tipo_via"),
        add_one_hot(base, "CLIMA", encoders["clima_categories"], "clima"),
        add_one_hot(base, "CARACTERISTICA_VIA", encoders["caracteristica_categories"], "caracteristica"),
        add_one_hot(base, "PERFIL_VIA", encoders["perfil_categories"], "perfil"),
        add_one_hot(base, "SUPERFICIE", encoders["superficie_categories"], "superficie"),
        _one_hot_combinations(base, "road_type_zone", encoders["road_type_zone_categories"], "road_type_zone"),
        _one_hot_combinations(base, "road_network_class", encoders["road_network_class_categories"], "road_network_class"),
    ]
    output = pd.concat([output, *one_hot_parts], axis=1)
    expected = encoders.get("feature_list")
    if expected is not None:
        for column in expected:
            if column not in output:
                output[column] = 0
        output = output[expected]
    return output.astype("float32").fillna(0)


def feature_availability_audit() -> pd.DataFrame:
    """Record availability claims supported by the retrospective extraction."""
    rows = [
        ("FECHA, HORA", "Consolidated registry", "timestamp not supplied", "not evaluated", "Calendar and cyclic-time features"),
        ("DEPARTAMENTO, coordenadas, red/tipo de vía", "Consolidated registry", "timestamp not supplied", "not evaluated", "Location and road context"),
        ("ZONA, CLIMA, geometría, superficie", "Consolidated registry", "timestamp not supplied", "not evaluated", "Scene and infrastructure context"),
        ("CLASE", "Consolidated registry", "timestamp not supplied", "no", "Retrospective classification"),
        ("VEHICULOS involucrados (conteos y tipo)", "VEHICULOS companion registry", "timestamp not supplied", "not evaluated", "retrospective historical classification", "include with scope restriction"),
        ("Todos los agregados de PERSONAS", "PERSONAS companion registry", "not usable: cardinality/deceased counts encode target", "yes", "none", "exclude entire source from predictors"),
        ("FALLECIDOS, LESIONADOS, VEHICULOS_DANADOS", "Outcome/count after event", "no", "no", "Excluded: direct outcome leakage"),
        ("CAUSA_FACTOR, CAUSA_ESPECIFICA", "Investigation conclusion", "no", "no", "Excluded: post-investigation leakage"),
        ("SENAL_VERTICAL, SENAL_HORIZONTAL", "Sparse recording field", "not reliable", "not reliable", "Excluded: missingness is period-dependent"),
    ]
    normalized = []
    for row in rows:
        if len(row) == 5:
            source_fields, source_stage, timestamp_evidence, target_dependency, decision = row
            normalized.append((source_fields, source_stage, timestamp_evidence, target_dependency, "retrospective historical classification", decision))
        else:
            normalized.append(row)
    return pd.DataFrame(normalized, columns=["source_fields", "source_stage", "timestamp_evidence", "target_dependency", "allowed_scope", "decision"])
