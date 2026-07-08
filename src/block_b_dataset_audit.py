from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "report" / "tables"

CSV_NAME = "Accidentes de tránsito en carreteras-2020-2021-Sutran.csv"
CSV_PATH = RAW_DIR / CSV_NAME

ENCODING = "latin-1"
SEPARATOR = ";"
SEED = 42


def normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.strip().upper()
    ascii_name = re.sub(r"[^A-Z0-9]+", "_", ascii_name)
    return ascii_name.strip("_")


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA})
    )


def replace_not_informed(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^\s*N\.?\s*I\.?\s*$", flags=re.IGNORECASE)
    return df.replace(pattern, np.nan, regex=True)


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
    return pd.read_csv(CSV_PATH, encoding=ENCODING, sep=SEPARATOR)


def audit_and_clean() -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    original_shape = raw.shape
    original_columns = list(raw.columns)
    not_informed_counts = {
        column: int(raw[column].astype("string").str.match(r"^\s*N\.?\s*I\.?\s*$", case=False, na=False).sum())
        for column in raw.columns
    }

    df = raw.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]
    normalized_columns = list(df.columns)
    df = replace_not_informed(df)

    for column in ["DEPARTAMENTO", "MODALIDAD", "CODIGO_VIA"]:
        if column in df.columns:
            df[column] = normalize_text(df[column])

    df["FECHA"] = pd.to_datetime(df["FECHA"].astype("string"), format="%Y%m%d", errors="coerce")
    fecha_parse_failures = int(df["FECHA"].isna().sum())

    hour_sample = (
        df["HORA"]
        .dropna()
        .sample(min(20, int(df["HORA"].dropna().shape[0])), random_state=SEED)
        .astype(str)
        .tolist()
    )
    df["hora_entera"] = df["HORA"].apply(parse_hour)
    invalid_hours = int(df["hora_entera"].isna().sum())

    for column in ["KILOMETRO", "FALLECIDOS", "HERIDOS"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] < 0, column] = np.nan

    kilometer_positive = df.loc[df["KILOMETRO"].notna() & (df["KILOMETRO"] >= 0), "KILOMETRO"]
    kilometer_upper_bound = float(kilometer_positive.quantile(0.995)) if not kilometer_positive.empty else np.nan
    kilometer_outliers = int((df["KILOMETRO"] > kilometer_upper_bound).sum()) if not np.isnan(kilometer_upper_bound) else 0
    if not np.isnan(kilometer_upper_bound):
        df.loc[df["KILOMETRO"] > kilometer_upper_bound, "KILOMETRO"] = np.nan

    outcome_upper_bounds: dict[str, float] = {}
    outcome_outliers: dict[str, int] = {}
    for column in ["FALLECIDOS", "HERIDOS"]:
        values = df[column].dropna()
        upper_bound = float(values.quantile(0.999)) if not values.empty else np.nan
        outcome_upper_bounds[column] = upper_bound
        outcome_outliers[column] = int((df[column] > upper_bound).sum()) if not np.isnan(upper_bound) else 0

    duplicates_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    rows_before_target_drop = int(df.shape[0])
    df = df[df["FALLECIDOS"].notna()].copy()
    target_nan_removed = rows_before_target_drop - int(df.shape[0])
    df["target_mortal"] = (df["FALLECIDOS"] > 0).astype("int8")

    if "FECHA_CORTE" in df.columns:
        df = df.drop(columns=["FECHA_CORTE"])

    final_shape = df.shape
    mortal_count = int(df["target_mortal"].sum())
    non_mortal_count = int((df["target_mortal"] == 0).sum())
    mortal_percentage = float(mortal_count / final_shape[0] * 100)

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
        "encoding": ENCODING,
        "separator": SEPARATOR,
        "raw_file": str(CSV_PATH.relative_to(ROOT)),
        "original_shape": list(original_shape),
        "final_shape": list(final_shape),
        "original_columns": original_columns,
        "normalized_columns": normalized_columns,
        "not_informed_counts": not_informed_counts,
        "fecha_min": df["FECHA"].min().date().isoformat(),
        "fecha_max": df["FECHA"].max().date().isoformat(),
        "fecha_parse_failures": fecha_parse_failures,
        "hour_sample": hour_sample,
        "invalid_hours_after_parse": invalid_hours,
        "kilometer_upper_bound_p995": kilometer_upper_bound,
        "kilometer_outliers_set_nan": kilometer_outliers,
        "outcome_upper_bounds_p999": outcome_upper_bounds,
        "outcome_outliers_above_p999": outcome_outliers,
        "duplicates_removed": duplicates_before,
        "target_nan_removed": target_nan_removed,
        "mortal_count": mortal_count,
        "non_mortal_count": non_mortal_count,
        "mortal_percentage": mortal_percentage,
        "contingency_size": "C2",
        "architecture": "Architecture A",
        "contingency_balance": "C5",
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
