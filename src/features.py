from __future__ import annotations

import math
import re
from typing import Any

import holidays
import numpy as np
import pandas as pd


SEED = 42

REGION_NATURAL_BY_DEPARTAMENTO = {
    "AMAZONAS": "SELVA",
    "ANCASH": "SIERRA",
    "APURIMAC": "SIERRA",
    "AREQUIPA": "COSTA",
    "AYACUCHO": "SIERRA",
    "CAJAMARCA": "SIERRA",
    "CALLAO": "COSTA",
    "CUSCO": "SIERRA",
    "HUANCAVELICA": "SIERRA",
    "HUANUCO": "SIERRA",
    "ICA": "COSTA",
    "JUNIN": "SIERRA",
    "LA LIBERTAD": "COSTA",
    "LAMBAYEQUE": "COSTA",
    "LIMA": "COSTA",
    "LORETO": "SELVA",
    "MADRE DE DIOS": "SELVA",
    "MOQUEGUA": "COSTA",
    "PASCO": "SIERRA",
    "PIURA": "COSTA",
    "PUNO": "SIERRA",
    "SAN MARTIN": "SELVA",
    "TACNA": "COSTA",
    "TUMBES": "COSTA",
    "UCAYALI": "SELVA",
}

REGION_CATEGORIES = ["COSTA", "SIERRA", "SELVA", "DESCONOCIDO"]
FRANJA_CATEGORIES = ["MADRUGADA", "MANANA", "TARDE", "NOCHE", "DESCONOCIDA"]
CLASE_CATEGORIES = ["ATROPELLO", "CHOQUE", "DESPISTE", "ESPECIAL", "VOLCADURA", "DESCONOCIDO"]
ZONA_CATEGORIES = ["RURAL", "URBANA", "DESCONOCIDO"]
RED_VIAL_CATEGORIES = ["NACIONAL", "DEPARTAMENTAL", "PROVINCIAL", "URBANO", "DESCONOCIDO"]
TIPO_VIA_CATEGORIES = ["CARRETERA", "AVENIDA", "CALLE", "EXPRESA", "OTRO", "DESCONOCIDO"]
CLIMA_CATEGORIES = ["DESPEJADO", "NUBLADO", "LLUVIOSO", "NIEBLA", "OTRO", "DESCONOCIDO"]
CARACTERISTICA_CATEGORIES = ["RECTO", "CURVA", "INTERSECCION", "ESTRUCTURA", "OTRO", "DESCONOCIDO"]
PERFIL_CATEGORIES = ["PLANA", "INCLINADA", "DESCONOCIDO"]
SUPERFICIE_CATEGORIES = ["PAVIMENTADA", "AFIRMADO", "TROCHA", "DESCONOCIDO"]


def normalize_text_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def extract_via_prefix(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    match = re.match(r"^([A-Z]+)", text)
    return match.group(1) if match else "DESCONOCIDO"


def normalize_clase(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    if text.startswith("ATROPELLO"):
        return "ATROPELLO"
    if text.startswith("CHOQUE"):
        return "CHOQUE"
    if text.startswith("DESPISTE"):
        return "DESPISTE"
    if text.startswith("VOLCADURA"):
        return "VOLCADURA"
    return "ESPECIAL"


def normalize_zona(value: Any) -> str:
    text = normalize_text_value(value)
    if text in {"RURAL", "URBANA"}:
        return text
    return "DESCONOCIDO"


def normalize_red_vial(value: Any) -> str:
    text = normalize_text_value(value)
    if text in {"NACIONAL", "DEPARTAMENTAL", "PROVINCIAL", "URBANO"}:
        return text
    return "DESCONOCIDO"


def normalize_tipo_via(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    if text == "CARRETERA":
        return "CARRETERA"
    if text == "AVENIDA":
        return "AVENIDA"
    if text in {"CALLE", "JIRÓN", "JIRON", "PASAJE", "ALAMEDA"}:
        return "CALLE"
    if text in {"VIA EXPRESA", "AUTOPISTA"}:
        return "EXPRESA"
    return "OTRO"


def normalize_clima(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    if text in {"DESPEJADO", "SOLEADO"}:
        return "DESPEJADO"
    if text in {"NUBLADO", "CIELO CUBIERTO", "PARCIALMENTE NUBLADO"}:
        return "NUBLADO"
    if text in {"LLUVIOSO", "GRANIZADO", "NEVADO"}:
        return "LLUVIOSO"
    if text in {"NIEBLA", "NEBLINA"}:
        return "NIEBLA"
    return "OTRO"


def normalize_caracteristica(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    if text == "TRAMO RECTO":
        return "RECTO"
    if text in {"CURVA", "SINUOSA"}:
        return "CURVA"
    if text in {"INTERSECCIÓN", "INTERSECCION", "ÓVALO", "OVALO"}:
        return "INTERSECCION"
    if text in {"PUENTE", "TÚNEL", "TUNEL", "PASE A DESNIVEL"}:
        return "ESTRUCTURA"
    return "OTRO"


def normalize_perfil(value: Any) -> str:
    text = normalize_text_value(value)
    if text in {"PLANA", "INCLINADA"}:
        return text
    return "DESCONOCIDO"


def normalize_superficie(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return "DESCONOCIDO"
    if text in {"ASFALTADA", "CONCRETO", "ADOQUINADO", "EMPEDRADO"}:
        return "PAVIMENTADA"
    if text == "AFIRMADO":
        return "AFIRMADO"
    if text in {"TROCHA", "CASCAJO/RIPIO"}:
        return "TROCHA"
    return "DESCONOCIDO"


def parse_hour_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    match_hhmm = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", text)
    if match_hhmm:
        hour = int(match_hhmm.group(1))
        return float(hour) if 0 <= hour <= 23 else np.nan

    digits = re.sub(r"\D", "", text)
    if digits:
        number = int(digits)
        hour = number // 100 if number >= 100 else number
        return float(hour) if 0 <= hour <= 23 else np.nan

    return np.nan


def assign_time_band(hour: float) -> str:
    if pd.isna(hour):
        return "DESCONOCIDA"
    hour_int = int(hour)
    if 0 <= hour_int <= 5:
        return "MADRUGADA"
    if 6 <= hour_int <= 11:
        return "MANANA"
    if 12 <= hour_int <= 17:
        return "TARDE"
    if 18 <= hour_int <= 23:
        return "NOCHE"
    return "DESCONOCIDA"


def add_one_hot(frame: pd.DataFrame, column: str, categories: list[str], prefix: str) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    values = frame[column].astype("string").fillna("DESCONOCIDO")
    for category in categories:
        safe_category = category.lower().replace(" ", "_")
        output[f"{prefix}_{safe_category}"] = (values == category).astype("int8")
    return output


def get_peru_holidays(years: pd.Series) -> set[pd.Timestamp]:
    valid_years = sorted({int(year) for year in years.dropna().unique()})
    calendar = holidays.country_holidays("PE", years=valid_years)
    return {pd.Timestamp(day) for day in calendar.keys()}


def derive_base_features(df: pd.DataFrame, via_frequency_map: dict[str, float] | None = None) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    fecha = pd.to_datetime(df["FECHA"], errors="coerce")
    years = fecha.dt.year
    holiday_days = get_peru_holidays(years)
    day = fecha.dt.day

    features["anio"] = years.astype("float64")
    features["mes"] = fecha.dt.month.astype("float64")
    features["dia_semana"] = fecha.dt.dayofweek.astype("float64")
    features["fin_de_semana"] = fecha.dt.dayofweek.isin([5, 6]).astype("int8")
    features["feriado"] = fecha.dt.normalize().isin(holiday_days).astype("int8")
    features["quincena"] = (day.between(14, 16) | (day >= 29) | (day <= 2)).astype("int8")

    hour = df["hora_entera"] if "hora_entera" in df.columns else df["HORA"].apply(parse_hour_value)
    hour = pd.to_numeric(hour, errors="coerce")
    features["hora_faltante"] = hour.isna().astype("int8")
    hour_filled = hour.fillna(0)
    features["hora_sin"] = np.where(hour.isna(), 0.0, np.sin(2 * math.pi * hour_filled / 24))
    features["hora_cos"] = np.where(hour.isna(), 0.0, np.cos(2 * math.pi * hour_filled / 24))
    features["nocturno"] = ((hour >= 20) | (hour <= 5)).fillna(False).astype("int8")
    features["franja"] = hour.apply(assign_time_band)

    departamento = df["DEPARTAMENTO"].map(normalize_text_value).fillna("DESCONOCIDO")
    features["DEPARTAMENTO"] = departamento
    features["region_natural"] = departamento.map(REGION_NATURAL_BY_DEPARTAMENTO).fillna("DESCONOCIDO")

    codigo_via = df["CODIGO_VIA"].map(normalize_text_value).fillna("DESCONOCIDO")
    features["CODIGO_VIA"] = codigo_via
    features["via_prefijo"] = codigo_via.apply(extract_via_prefix)
    if via_frequency_map is None:
        features["via_freq"] = 0.0
    else:
        features["via_freq"] = codigo_via.map(via_frequency_map).fillna(0.0).astype("float64")

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
    return features
