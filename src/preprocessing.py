from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
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
    add_one_hot,
    derive_base_features,
)


CONTINUOUS_COLUMNS = ["LATITUD", "LONGITUD", "via_freq", "anio"]


def fit_preprocessing_artifacts(train_df: pd.DataFrame) -> tuple[StandardScaler, dict[str, Any]]:
    train_base = derive_base_features(train_df)

    lat_median = float(train_base["LATITUD"].median())
    lon_median = float(train_base["LONGITUD"].median())
    via_frequency_map = (
        train_base["CODIGO_VIA"]
        .value_counts(normalize=True, dropna=False)
        .astype(float)
        .to_dict()
    )

    train_base = derive_base_features(train_df, via_frequency_map=via_frequency_map)
    train_base["LATITUD"] = train_base["LATITUD"].fillna(lat_median)
    train_base["LONGITUD"] = train_base["LONGITUD"].fillna(lon_median)

    scaler = StandardScaler()
    scaler.fit(train_base[CONTINUOUS_COLUMNS])

    encoders: dict[str, Any] = {
        "lat_median": lat_median,
        "lon_median": lon_median,
        "via_frequency_map": via_frequency_map,
        "departamento_categories": sorted(train_base["DEPARTAMENTO"].dropna().unique().tolist()),
        "via_prefijo_categories": sorted(train_base["via_prefijo"].dropna().unique().tolist()),
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
        "continuous_columns": CONTINUOUS_COLUMNS,
    }

    train_features = preparar_entrada(train_df, scaler=scaler, encoders=encoders)
    encoders["feature_list"] = train_features.columns.tolist()
    return scaler, encoders


def preparar_entrada(df: pd.DataFrame, scaler: StandardScaler, encoders: dict[str, Any]) -> pd.DataFrame:
    base = derive_base_features(df, via_frequency_map=encoders["via_frequency_map"])
    base["LATITUD"] = base["LATITUD"].fillna(float(encoders["lat_median"]))
    base["LONGITUD"] = base["LONGITUD"].fillna(float(encoders["lon_median"]))

    output = pd.DataFrame(index=base.index)
    continuous_columns = list(encoders["continuous_columns"])
    scaled = scaler.transform(base[continuous_columns])
    for index, column in enumerate(continuous_columns):
        output[column] = scaled[:, index]

    passthrough_columns = [
        "mes",
        "dia_semana",
        "fin_de_semana",
        "feriado",
        "quincena",
        "hora_faltante",
        "hora_sin",
        "hora_cos",
        "nocturno",
        "coord_faltante",
    ]
    for column in passthrough_columns:
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
    ]
    output = pd.concat([output, *one_hot_parts], axis=1)

    feature_list = encoders.get("feature_list")
    if feature_list is not None:
        for column in feature_list:
            if column not in output.columns:
                output[column] = 0
        output = output[feature_list]

    return output.fillna(0)


def save_artifacts(models_dir: Path, scaler: StandardScaler, encoders: dict[str, Any]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(encoders, models_dir / "encoders.pkl")


def load_artifacts(models_dir: Path) -> tuple[StandardScaler, dict[str, Any]]:
    scaler = joblib.load(models_dir / "scaler.pkl")
    encoders = joblib.load(models_dir / "encoders.pkl")
    return scaler, encoders
