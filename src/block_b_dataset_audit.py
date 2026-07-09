from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "report" / "tables"

XLSX_NAME = "BBDD_ONSV_SINIESTROS_FATALES_2021-2025.xlsx"
XLSX_PATH = RAW_DIR / XLSX_NAME
SHEET_NAME = "SINIESTROS"
HEADER_ROW = 4

SEED = 42

# ONSV column -> canonical column. Post-outcome columns (lesionados, vehiculos
# danados, causa) are kept in the clean base for EDA but excluded from features.
COLUMN_MAP = {
    "CÓDIGO SINIESTRO": "CODIGO_SINIESTRO",
    "FECHA SINIESTRO": "FECHA",
    "HORA SINIESTRO": "HORA",
    "CLASE SINIESTRO": "CLASE",
    "CANTIDAD DE FALLECIDOS": "FALLECIDOS",
    "CANTIDAD DE LESIONADOS": "LESIONADOS",
    "CANTIDAD DE VEHICULOS DAÑADOS": "VEHICULOS_DANADOS",
    "DEPARTAMENTO": "DEPARTAMENTO",
    "PROVINCIA": "PROVINCIA",
    "DISTRITO": "DISTRITO",
    "ZONA": "ZONA",
    "TIPO DE VÍA": "TIPO_VIA",
    "RED VIAL": "RED_VIAL",
    "COD CARRETERA": "CODIGO_VIA",
    "COORDENADAS LATITUD": "LATITUD",
    "COORDENADAS  LONGITUD": "LONGITUD",
    "CONDICIÓN CLIMÁTICA": "CLIMA",
    "ZONIFICACIÓN": "ZONIFICACION",
    "CARACTERÍSTICAS DE VÍA": "CARACTERISTICA_VIA",
    "PERFIL LONGITUDINAL VÍA": "PERFIL_VIA",
    "SUPERFICIE DE CALZADA": "SUPERFICIE",
    "¿EXISTE SEÑAL VERTICAL?": "SENAL_VERTICAL",
    "¿EXISTE SEÑAL HORIZONTAL?": "SENAL_HORIZONTAL",
    "CAUSA FACTOR PRINCIPAL": "CAUSA_FACTOR",
    "CAUSA ESPECÍFICA": "CAUSA_ESPECIFICA",
}

# Peru bounding box used to null out impossible coordinates.
LAT_BOUNDS = (-18.4, 0.1)
LON_BOUNDS = (-81.4, -68.6)

MISSING_TOKENS = {"", "SIN INFO", "NO CORRESPONDE", "SIN CLASIFICAR", "-"}


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.replace({token: pd.NA for token in MISSING_TOKENS})


def parse_hour(value: Any) -> float:
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


def load_raw_dataset() -> pd.DataFrame:
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW, dtype=str)
    raw.columns = [str(column).strip() for column in raw.columns]
    return raw


def audit_and_clean() -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    original_shape = raw.shape
    original_columns = list(raw.columns)

    df = raw.rename(columns=COLUMN_MAP)
    df = df[[column for column in COLUMN_MAP.values() if column in df.columns]]
    df = df.dropna(subset=["CODIGO_SINIESTRO"]).copy()
    header_noise_removed = int(original_shape[0] - df.shape[0])

    text_columns = [
        "CLASE",
        "DEPARTAMENTO",
        "PROVINCIA",
        "DISTRITO",
        "ZONA",
        "TIPO_VIA",
        "RED_VIAL",
        "CODIGO_VIA",
        "CLIMA",
        "ZONIFICACION",
        "CARACTERISTICA_VIA",
        "PERFIL_VIA",
        "SUPERFICIE",
        "SENAL_VERTICAL",
        "SENAL_HORIZONTAL",
        "CAUSA_FACTOR",
        "CAUSA_ESPECIFICA",
    ]
    for column in text_columns:
        df[column] = normalize_text(df[column])

    df["FECHA"] = pd.to_datetime(df["FECHA"].astype("string"), dayfirst=True, errors="coerce")
    fecha_parse_failures = int(df["FECHA"].isna().sum())
    df = df[df["FECHA"].notna()].copy()

    df["hora_entera"] = df["HORA"].apply(parse_hour)
    invalid_hours = int(df["hora_entera"].isna().sum())

    for column in ["FALLECIDOS", "LESIONADOS", "VEHICULOS_DANADOS"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] < 0, column] = np.nan

    for column, bounds in [("LATITUD", LAT_BOUNDS), ("LONGITUD", LON_BOUNDS)]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        out_of_bounds = ~df[column].between(*bounds)
        df.loc[out_of_bounds, column] = np.nan
    coordinate_missing = int((df["LATITUD"].isna() | df["LONGITUD"].isna()).sum())

    duplicates_before = int(df.duplicated(subset=["CODIGO_SINIESTRO"]).sum())
    df = df.drop_duplicates(subset=["CODIGO_SINIESTRO"]).reset_index(drop=True)

    rows_before_target_drop = int(df.shape[0])
    df = df[df["FALLECIDOS"].notna() & (df["FALLECIDOS"] >= 1)].copy()
    target_invalid_removed = rows_before_target_drop - int(df.shape[0])

    # Reframed target: within fatal crashes, flag multiple-fatality events.
    df["target_multifatal"] = (df["FALLECIDOS"] >= 2).astype("int8")

    final_shape = df.shape
    multifatal_count = int(df["target_multifatal"].sum())
    single_fatal_count = int((df["target_multifatal"] == 0).sum())
    multifatal_percentage = float(multifatal_count / final_shape[0] * 100)

    missing_summary = (
        df.isna()
        .mean()
        .mul(100)
        .rename("missing_percentage")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_summary.to_csv(TABLES_DIR / "tab01_missing_values_block_b.csv", index=False)

    audit_summary = {
        "raw_file": str(XLSX_PATH.relative_to(ROOT)),
        "sheet": SHEET_NAME,
        "header_row": HEADER_ROW,
        "original_shape": list(original_shape),
        "final_shape": list(final_shape),
        "original_columns": original_columns,
        "canonical_columns": list(df.columns),
        "header_noise_removed": header_noise_removed,
        "fecha_min": df["FECHA"].min().date().isoformat(),
        "fecha_max": df["FECHA"].max().date().isoformat(),
        "fecha_parse_failures": fecha_parse_failures,
        "invalid_hours_after_parse": invalid_hours,
        "coordinate_rows_out_of_bounds_or_missing": coordinate_missing,
        "duplicates_removed": duplicates_before,
        "target_invalid_removed": target_invalid_removed,
        "multifatal_count": multifatal_count,
        "single_fatal_count": single_fatal_count,
        "multifatal_percentage": multifatal_percentage,
        "target_definition": "target_multifatal = 1 si FALLECIDOS >= 2, 0 si FALLECIDOS == 1",
        "leakage_exclusions": [
            "LESIONADOS",
            "VEHICULOS_DANADOS",
            "CAUSA_FACTOR",
            "CAUSA_ESPECIFICA",
            "SENAL_VERTICAL",
            "SENAL_HORIZONTAL",
        ],
        "leakage_rationale": (
            "LESIONADOS y VEHICULOS_DANADOS se cuentan despues del siniestro; "
            "CAUSA se determina en la investigacion posterior (49% en proceso); "
            "las senales tienen 78% de faltantes concentrados por periodo de registro."
        ),
        "balance_strategy": "class_weight only",
    }

    pd.DataFrame([audit_summary]).to_csv(TABLES_DIR / "tab00_audit_summary_block_b.csv", index=False)
    (TABLES_DIR / "tab00_audit_summary_block_b.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    df.to_parquet(PROCESSED_DIR / "base_limpia.parquet", index=False)
    return audit_summary


if __name__ == "__main__":
    summary = audit_and_clean()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
