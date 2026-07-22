"""Generate five deterministic, hash-sealed academic demonstrations from 2023."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.chronology import CALIBRATION_THRESHOLD_VALIDATION_PARTITION
from src.final_model_bundle import CanonicalModelBundle, sha256_file
BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"
DEMO_PATH = ROOT / "data" / "processed" / "demo_cases.csv"
DEMO_MANIFEST_PATH = ROOT / "data" / "processed" / "demo_cases_manifest.json"
FINAL_DIR = ROOT / "models" / "final"


ROLE_ORDER = ["TN", "FP", "boundary_FN", "TP", "synthetic_unseen_code"]
DESCRIPTIONS = {
    "TN": "TN real 2023 · clase 1 correctamente estimada",
    "FP": "FP real 2023 · alerta multifatal no observada",
    "boundary_FN": "FN limítrofe real 2023 · multifatal debajo del umbral",
    "TP": "TP real 2023 · multifatal correctamente estimado",
    "synthetic_unseen_code": "Sensibilidad sintética · solo cambia un código de vía no visto",
}


def _pick(frame: pd.DataFrame, role: str, threshold: float) -> pd.Series:
    actual = frame["actual_multifatal"]
    predicted = frame["calibrated_probability"].ge(threshold).astype(int)
    if role == "TN":
        candidates = frame[(actual == 0) & (predicted == 0)].sort_values(
            ["calibrated_probability", "source_index"]
        )
    elif role == "FP":
        candidates = frame[(actual == 0) & (predicted == 1)].assign(
            distance=lambda value: value["calibrated_probability"] - threshold
        ).sort_values(["distance", "source_index"])
    elif role == "boundary_FN":
        candidates = frame[(actual == 1) & (predicted == 0)].assign(
            distance=lambda value: threshold - value["calibrated_probability"]
        ).sort_values(["distance", "source_index"])
    elif role == "TP":
        candidates = frame[(actual == 1) & (predicted == 1)].sort_values(
            ["calibrated_probability", "source_index"], ascending=[False, True]
        )
    else:
        raise ValueError(role)
    if candidates.empty:
        raise RuntimeError(f"No 2023 candidate satisfies demonstration role {role}.")
    return candidates.iloc[0]


def generate_demo_cases(root: Path = ROOT) -> dict[str, object]:
    base_path = root / "data" / "processed" / "base_limpia.parquet"
    final_dir = root / "models" / "final"
    destination = root / "data" / "processed" / "demo_cases.csv"
    manifest_path = root / "data" / "processed" / "demo_cases_manifest.json"
    base = pd.read_parquet(base_path, filters=[("FECHA", ">=", pd.Timestamp("2023-01-01")), ("FECHA", "<", pd.Timestamp("2024-01-01"))])
    if base.empty or not pd.to_datetime(base["FECHA"]).dt.year.eq(2023).all():
        raise RuntimeError("Demo sourcing is restricted to 2023 before materialization.")
    runtime = CanonicalModelBundle(final_dir, verify_hashes=True)
    predictions = runtime.predict_dataframe(base)
    scored = base.copy()
    scored["source_index"] = base.index
    scored["actual_multifatal"] = base["target_multifatal"].astype(int)
    scored["raw_probability"] = predictions["raw_probability"].to_numpy()
    scored["calibrated_probability"] = predictions["calibrated_probability"].to_numpy()
    threshold = float(runtime.thresholds["calibrated"]["value"])
    rows: list[pd.Series] = []
    for role in ROLE_ORDER[:4]:
        row = _pick(scored, role, threshold).copy()
        row["role"] = role
        row["descripcion"] = DESCRIPTIONS[role]
        rows.append(row)
    synthetic = rows[0].copy()
    source_probability = float(synthetic["calibrated_probability"])
    synthetic["CODIGO_VIA"] = "ZZ-UNSEEN"
    synthetic_input = pd.DataFrame([synthetic])
    synthetic_prediction = runtime.predict_dataframe(synthetic_input).iloc[0]
    synthetic["role"] = "synthetic_unseen_code"
    synthetic["descripcion"] = DESCRIPTIONS["synthetic_unseen_code"]
    synthetic["actual_multifatal"] = np.nan
    synthetic["raw_probability"] = float(synthetic_prediction["raw_probability"])
    synthetic["calibrated_probability"] = float(synthetic_prediction["calibrated_probability"])
    synthetic["source_expected_calibrated_probability"] = source_probability
    rows.append(synthetic)

    schema = json.loads((final_dir / "feature_schema.json").read_text(encoding="utf-8"))
    input_fields = [str(item["name"]) for item in schema["required_raw_fields"]]
    output = pd.DataFrame(rows)
    output.insert(0, "caso_id", ["demo_01_tn", "demo_02_fp", "demo_03_boundary_fn", "demo_04_tp", "demo_05_unseen_code"])
    output["FECHA"] = pd.to_datetime(output["FECHA"]).dt.strftime("%Y-%m-%d")
    output["expected_raw_probability"] = output["raw_probability"].astype(float)
    output["expected_calibrated_probability"] = output["calibrated_probability"].astype(float)
    output["expected_calibrated_prediction"] = output["expected_calibrated_probability"].ge(threshold).astype(int)
    output["truth_status"] = np.where(output["role"].eq("synthetic_unseen_code"), "unavailable_synthetic", "observed_2023")
    if "source_expected_calibrated_probability" not in output:
        output["source_expected_calibrated_probability"] = np.nan
    columns = [
        "caso_id", "role", "descripcion", "truth_status", "source_index",
        "actual_multifatal", *input_fields, "expected_raw_probability",
        "expected_calibrated_probability", "expected_calibrated_prediction",
        "source_expected_calibrated_probability",
    ]
    output = output[columns]
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, float_format="%.10g")
    payload = {
        "schema_version": 1,
        "model_version": runtime.manifest["model_version"],
        "calibration_threshold_validation_partition": CALIBRATION_THRESHOLD_VALIDATION_PARTITION,
        "reference_2024_2025_loaded": False,
        "roles_in_order": ROLE_ORDER,
        "visible_decision_scale": "calibrated",
        "visible_threshold": threshold,
        "synthetic_change": "CODIGO_VIA only; observed truth unavailable; sensitivity is not causality",
        "csv_path": str(destination.relative_to(root)),
        "csv_sha256": sha256_file(destination),
        "model_sha256": runtime.manifest["artifact_hashes"]["model.keras"],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    canonical_manifest_path = final_dir / "manifest.json"
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    canonical_manifest["demo_artifacts"] = {
        str(destination.relative_to(root)): sha256_file(destination),
        str(manifest_path.relative_to(root)): sha256_file(manifest_path),
    }
    canonical_manifest_path.write_text(json.dumps(canonical_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(generate_demo_cases(), ensure_ascii=False, indent=2))
