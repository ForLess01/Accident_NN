from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app_inference import historical_comparison, load_demo_cases, predict_records


TABLES_DIR = ROOT / "report" / "tables"


def run_block_g_check() -> dict[str, object]:
    demo_cases = load_demo_cases()
    start = time.perf_counter()
    predictions = predict_records(demo_cases)
    elapsed = time.perf_counter() - start

    base = predictions.loc[predictions["caso_id"] == "demo_01_tipico_no_mortal", "probabilidad_mortal"].iloc[0]
    atropello = predictions.loc[predictions["caso_id"] == "demo_02_atropello", "probabilidad_mortal"].iloc[0]
    unseen = predictions.loc[predictions["caso_id"] == "demo_04_codigo_no_visto", "probabilidad_mortal"].iloc[0]
    puno = predictions.loc[predictions["caso_id"] == "demo_05_puno", "probabilidad_mortal"].iloc[0]
    base_score = predictions.loc[predictions["caso_id"] == "demo_01_tipico_no_mortal", "score_riesgo_mortal"].iloc[0]
    calibration_method = predictions["calibracion"].iloc[0]
    comparison = historical_comparison(demo_cases.iloc[0], float(base_score))
    historical_count = int(comparison["tasa_mortalidad"].notna().sum())

    checklist = [
        {"check": "demo_cases_predict", "passed": True, "detail": f"{len(predictions)} casos predichos"},
        {"check": "response_under_2_seconds", "passed": bool(elapsed < 2.0), "detail": f"{elapsed:.3f}s"},
        {"check": "atropello_changes_probability", "passed": bool(float(atropello) != float(base)), "detail": f"base={base:.4f}; atropello={atropello:.4f}"},
        {"check": "unseen_road_code_does_not_break", "passed": bool(pd.notna(unseen)), "detail": f"probabilidad={unseen:.4f}"},
        {"check": "puno_case_does_not_break", "passed": bool(pd.notna(puno)), "detail": f"probabilidad={puno:.4f}"},
        {
            "check": "historical_comparison_available",
            "passed": bool(historical_count >= 4),
            "detail": f"{historical_count} tasas disponibles",
        },
        {
            "check": "posthoc_calibration_available",
            "passed": bool(pd.notna(base_score) and calibration_method != "sin_calibrador"),
            "detail": f"score={base_score:.4f}; calibrador={calibration_method}",
        },
    ]
    table = pd.DataFrame(checklist)
    table.to_csv(TABLES_DIR / "tab09_gui_checklist.csv", index=False)
    predictions.to_csv(TABLES_DIR / "tab09_gui_demo_predictions.csv", index=False)

    summary = {
        "passed": bool(table["passed"].all()),
        "elapsed_seconds": float(elapsed),
        "checks": checklist,
    }
    (TABLES_DIR / "tab09_gui_checklist_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_block_g_check(), ensure_ascii=False, indent=2))
