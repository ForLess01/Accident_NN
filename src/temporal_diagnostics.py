"""Nested rolling temporal diagnostics using only rows through 2023."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_e_modeling import choose_threshold, evaluate, train_mlp_grid
from src.final_model_bundle import apply_calibrator, select_calibrator_validation_only
from src.model_protocol import EXCLUDED_COLUMNS, fit_preprocessor, transform_features


BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
TABLES_DIR = ROOT / "report" / "tables"


FOLDS = (
    {
        "fold": "rolling_1",
        "fit": ("2021-01-01", "2021-06-30"),
        "selection": ("2021-07-01", "2021-12-31"),
        "calibration": ("2022-01-01", "2022-06-30"),
        "outer": ("2022-07-01", "2022-12-31"),
    },
    {
        "fold": "rolling_2",
        "fit": ("2021-01-01", "2021-12-31"),
        "selection": ("2022-01-01", "2022-06-30"),
        "calibration": ("2022-07-01", "2022-12-31"),
        "outer": ("2023-01-01", "2023-12-31"),
    },
)


def _slice(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(frame["FECHA"])
    return frame.loc[dates.between(*map(pd.Timestamp, bounds))].copy()


def _xy(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return (
        part.drop(columns=[column for column in EXCLUDED_COLUMNS if column in part]),
        part["target_multifatal"].astype("int8"),
    )


def run(root: Path = ROOT) -> dict[str, object]:
    source = pd.read_parquet(
        root / "data" / "processed" / "base_limpia.parquet",
        filters=[("FECHA", "<", pd.Timestamp("2024-01-01"))],
    )
    source["FECHA"] = pd.to_datetime(source["FECHA"], errors="raise")
    if source["FECHA"].dt.year.max() > 2023:
        raise RuntimeError("Nested rolling diagnostics may not materialize the historical reference.")
    rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    for definition in FOLDS:
        parts = {role: _slice(source, definition[role]) for role in ("fit", "selection", "calibration", "outer")}
        for role, part in parts.items():
            if part.empty:
                raise RuntimeError(f"{definition['fold']} {role} is empty.")
            role_rows.append({
                "fold": definition["fold"], "role": role,
                "date_min": part["FECHA"].min().date().isoformat(),
                "date_max": part["FECHA"].max().date().isoformat(),
                "n": len(part), "positives": int(part["target_multifatal"].sum()),
            })
        ordered = [parts[role]["FECHA"] for role in ("fit", "selection", "calibration", "outer")]
        if not all(left.max() < right.min() for left, right in zip(ordered, ordered[1:])):
            raise RuntimeError(f"{definition['fold']} roles overlap or are not chronological.")
        X_fit_raw, y_fit = _xy(parts["fit"])
        X_select_raw, y_select = _xy(parts["selection"])
        X_cal_raw, y_cal = _xy(parts["calibration"])
        X_outer_raw, y_outer = _xy(parts["outer"])
        scaler, encoders = fit_preprocessor(X_fit_raw)
        X_fit = transform_features(X_fit_raw, scaler, encoders)
        X_select = transform_features(X_select_raw, scaler, encoders)
        X_cal = transform_features(X_cal_raw, scaler, encoders)
        X_outer = transform_features(X_outer_raw, scaler, encoders)
        model, config, seed, _, _, _, _ = train_mlp_grid(X_fit, y_fit, X_select, y_select)
        raw_cal = model.predict(X_cal, verbose=0).reshape(-1)
        method, calibrator, calibration, oof_calibrated = select_calibrator_validation_only(raw_cal, y_cal)
        threshold = float(choose_threshold(y_cal, oof_calibrated)["threshold"])
        raw_outer = model.predict(X_outer, verbose=0).reshape(-1)
        calibrated_outer = apply_calibrator(calibrator, method, raw_outer)
        rows.append({
            "fold": definition["fold"], "config_id": config.config_id, "seed": seed,
            "calibration_method": method, "calibrated_threshold": threshold,
            **evaluate(y_outer, calibrated_outer, threshold),
        })
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    roles = pd.DataFrame(role_rows)
    metrics = pd.DataFrame(rows)
    roles.to_csv(TABLES_DIR / "temporal_nested_fold_roles.csv", index=False)
    metrics.to_csv(TABLES_DIR / "temporal_nested_outer_metrics.csv", index=False)
    payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "diagnostic estimate of temporal variability; not a new untouched test",
        "source_boundary": "FECHA < 2024-01-01 enforced before materialization",
        "within_fold_roles_disjoint": True,
        "folds": rows,
        "limitations": "Only two rolling folds are possible with annual 2021-2023 data; folds share historical rows and do not replace external/prospective validation.",
    }
    (TABLES_DIR / "temporal_nested_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
