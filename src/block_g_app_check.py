"""Focused, read-only checks for the canonical Streamlit application."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.app_inference as app_inference
from src.app_inference import (
    load_demo_cases,
    load_feature_schema,
    load_manifest,
    load_prediction_stack,
    load_thresholds,
    predict_records,
)


def run_block_g_check(run_apptest: bool = True) -> dict[str, object]:
    """Validate the deployed contract without writing or rebuilding artifacts."""
    manifest = load_manifest()
    schema = load_feature_schema()
    thresholds = load_thresholds()
    runtime = load_prediction_stack()
    demos = load_demo_cases()
    if demos.empty:
        raise RuntimeError("No demo records are available for the app check.")

    start = time.perf_counter()
    predictions = predict_records(demos)
    elapsed = time.perf_counter() - start
    numeric = predictions[["raw_probability", "calibrated_probability", "raw_threshold", "calibrated_threshold"]]
    source = inspect.getsource(app_inference)
    app_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")

    checklist: list[dict[str, object]] = [
        {
            "check": "canonical_manifest_loaded",
            "passed": manifest["model_version"] == "canonical-3.0.0",
            "detail": str(manifest["model_version"]),
        },
        {
            "check": "model_schema_feature_alignment",
            "passed": int(runtime.model.input_shape[-1]) == int(schema["processed_feature_count"]) == int(manifest["feature_count"]),
            "detail": f'model={runtime.model.input_shape[-1]}; schema={schema["processed_feature_count"]}',
        },
        {
            "check": "finite_runtime_outputs",
            "passed": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
            "detail": f"{len(predictions)} records",
        },
        {
            "check": "probability_scales_are_explicit",
            "passed": bool(
                float(thresholds["raw"]["value"]) == float(manifest["thresholds"]["raw"]["value"])
                and float(thresholds["calibrated"]["value"]) == float(manifest["thresholds"]["calibrated"]["value"])
                and "raw_probability" in predictions
                and "calibrated_probability" in predictions
            ),
            "detail": f'raw={thresholds["raw"]["value"]}; calibrated={thresholds["calibrated"]["value"]}',
        },
        {
            "check": "runtime_is_read_only",
            "passed": not any(token in source for token in ("audit_and_clean", "to_csv(", "to_parquet(", "joblib.dump")),
            "detail": "No training, preprocessing build, or writes in src.app_inference",
        },
        {
            "check": "schema_drives_inputs",
            "passed": "load_input_options" in app_source and "st.form(" in app_source and "st.time_input(" in app_source,
            "detail": "Canonical categories + explicit form submission + time control",
        },
        {
            "check": "canonical_evidence_only",
            "passed": "final_reference_metrics_2024_2025.json" in app_source and "load_explainability_artifacts" in app_source,
            "detail": "UI reads only definitive reference tables and rejects obsolete evidence",
        },
        {
            "check": "warm_prediction_under_2_seconds",
            "passed": elapsed < 2.0,
            "detail": f"{elapsed:.3f}s for {len(demos)} records after bundle load",
        },
    ]

    if run_apptest:
        try:
            from streamlit.testing.v1 import AppTest
        except (ImportError, ModuleNotFoundError) as exc:
            checklist.append(
                {
                    "check": "streamlit_apptest_smoke",
                    "passed": False,
                    "detail": f"Required AppTest unavailable: {exc}",
                }
            )
        else:
            app_path = str(ROOT / "app" / "streamlit_app.py")
            sections = {
                "panorama": "Panorama",
                "estimar": "Probar la red neuronal",
                "explorar": "Explorar datos",
                "regiones": "Patrones regionales",
                "evidencia": "Evidencia del modelo",
            }
            navigation_runs: dict[str, dict[str, object]] = {}
            for slug, heading in sections.items():
                page = AppTest.from_file(app_path, default_timeout=45)
                page.query_params["section"] = slug
                page.run()
                navigation_runs[slug] = {
                    "heading": any(item.value == heading for item in page.header),
                    "errors": [str(item.value) for item in page.error],
                    "exceptions": [str(item.value) for item in page.exception],
                    "radio": [str(item.value) for item in page.radio],
                    "expanders": [str(item.label) for item in page.expander],
                }

            test = AppTest.from_file(app_path, default_timeout=45)
            test.query_params["section"] = "estimar"
            test.run()
            empty_defaults = bool(
                test.date_input[0].value is None
                and test.time_input[0].value is None
                and test.selectbox[0].value is None
                and test.number_input[0].value is None
                and test.number_input[1].value is None
                and not any(item.value == "Resultado" for item in test.subheader)
            )
            demo_runs: dict[str, dict[str, object]] = {}
            for demo_id in demos["caso_id"].astype(str):
                demo_test = AppTest.from_file(app_path, default_timeout=45)
                demo_test.query_params["section"] = "estimar"
                demo_test.run()
                demo_test.selectbox[0].set_value(demo_id)
                demo_test.button[0].click().run()
                demo_row = demos.loc[demos["caso_id"].eq(demo_id)].iloc[0]
                normalized_code = app_inference.normalize_road_code(demo_row["CODIGO_VIA"])
                expected_road_code = None if normalized_code == "DESCONOCIDO" else normalized_code
                loaded = bool(
                    demo_test.date_input[0].value is not None
                    and demo_test.time_input[0].value is not None
                    and demo_test.selectbox[1].value is not None
                    and demo_test.selectbox[2].value == expected_road_code
                    and demo_test.number_input[0].value is not None
                    and demo_test.number_input[1].value is not None
                )
                load_errors = [str(item.value) for item in demo_test.error]
                load_exceptions = [str(item.value) for item in demo_test.exception]
                warnings = [str(item.value) for item in demo_test.warning]
                unseen_warning = demo_row["role"] != "synthetic_unseen_code" or any(
                    "no observado en entrenamiento" in item.lower() for item in warnings
                )
                demo_test.button[1].click().run()
                submit_errors = [str(item.value) for item in demo_test.error]
                submit_exceptions = [str(item.value) for item in demo_test.exception]
                result_rendered = any(item.value == "Resultado" for item in demo_test.subheader)
                info_messages = [str(item.value) for item in demo_test.info]
                provenance_rendered = (
                    any("Sensibilidad sintética" in item for item in info_messages)
                    if demo_row["role"] == "synthetic_unseen_code"
                    else any(f"Rol pedagógico {demo_row['role']}" in item for item in info_messages)
                )
                demo_runs[demo_id] = {
                    "loaded": loaded,
                    "unseen_warning": unseen_warning,
                    "load_errors": load_errors,
                    "load_exceptions": load_exceptions,
                    "submit_errors": submit_errors,
                    "submit_exceptions": submit_exceptions,
                    "result": result_rendered,
                    "provenance": provenance_rendered,
                }

            demos_ok = all(
                bool(result["loaded"])
                and bool(result["unseen_warning"])
                and not result["load_errors"]
                and not result["load_exceptions"]
                and not result["submit_errors"]
                and not result["submit_exceptions"]
                and bool(result["result"])
                and bool(result["provenance"])
                for result in demo_runs.values()
            )

            def edited_demo_semantics(demo_id: str) -> dict[str, object]:
                edited = AppTest.from_file(app_path, default_timeout=45)
                edited.query_params["section"] = "estimar"
                edited.run()
                edited.selectbox[0].set_value(demo_id)
                edited.button[0].click().run()
                total = next(item for item in edited.number_input if item.label == "Vehículos involucrados")
                total.set_value(int(total.value) + 1).run()
                stale_before_submit = any(item.value == "Resultado" for item in edited.subheader)
                edited.button[1].click().run()
                messages = [str(item.value) for item in edited.info]
                return {
                    "errors": [str(item.value) for item in edited.error],
                    "exceptions": [str(item.value) for item in edited.exception],
                    "result": any(item.value == "Resultado" for item in edited.subheader),
                    "stale_before_submit": stale_before_submit,
                    "edited_notice": any("escenario fue editado" in item.lower() for item in messages),
                    "leaked_role_or_truth": any(
                        "Rol pedagógico" in item or "Observado en 2023" in item or "Sensibilidad sintética" in item
                        for item in messages
                    ),
                }

            edited_real = edited_demo_semantics("demo_01_tn")
            edited_synthetic = edited_demo_semantics("demo_05_unseen_code")
            edited_semantics_ok = all(
                not result["errors"]
                and not result["exceptions"]
                and bool(result["result"])
                and not bool(result["stale_before_submit"])
                and bool(result["edited_notice"])
                and not bool(result["leaked_role_or_truth"])
                for result in (edited_real, edited_synthetic)
            )

            # Reuse a clean canonical demo for the stale-result regression.
            test.selectbox[0].set_value("demo_01_tn")
            test.button[0].click().run()
            test.button[1].click().run()
            valid_errors = [str(item.value) for item in test.error]
            valid_exceptions = [str(item.value) for item in test.exception]
            valid_result_rendered = any(item.value == "Resultado" for item in test.subheader)

            # Regression: a department/coordinate mismatch clears the prior
            # decision and reports a field-level coherence error.
            test.selectbox[1].set_value("PUNO")
            test.button[1].click().run()
            invalid_errors = [str(item.value) for item in test.error]
            invalid_exceptions = [str(item.value) for item in test.exception]
            stale_result_rendered = any(item.value == "Resultado" for item in test.subheader)
            validation_error_rendered = any("no corresponden" in item.lower() for item in invalid_errors)

            evidence_expanders = navigation_runs["evidencia"]["expanders"]
            table_fallbacks_present = bool(
                any("matriz de confusión" in label.lower() for label in evidence_expanders)
                and any("curvas pr y roc" in label.lower() for label in evidence_expanders)
            )
            navigation_ok = all(
                bool(result["heading"]) and not result["errors"] and not result["exceptions"]
                for result in navigation_runs.values()
            )
            app_ok = bool(
                navigation_ok
                and empty_defaults
                and demos_ok
                and edited_semantics_ok
                and not valid_errors
                and not valid_exceptions
                and valid_result_rendered
                and validation_error_rendered
                and not invalid_exceptions
                and not stale_result_rendered
                and table_fallbacks_present
            )
            checklist.append(
                {
                    "check": "streamlit_apptest_smoke",
                    "passed": app_ok,
                    "detail": (
                        "All 5 URL sections and all 5 canonical demos passed; edited real/synthetic provenance was cleared; mismatch stale-state guard + chart tables passed"
                        if app_ok
                        else json.dumps(
                            {
                                "navigation": navigation_runs,
                                "empty_defaults": empty_defaults,
                                "demo_runs": demo_runs,
                                "edited_real": edited_real,
                                "edited_synthetic": edited_synthetic,
                                "valid_errors": valid_errors,
                                "valid_exceptions": valid_exceptions,
                                "valid_result": valid_result_rendered,
                                "invalid_errors": invalid_errors,
                                "invalid_exceptions": invalid_exceptions,
                                "stale_result": stale_result_rendered,
                                "table_fallbacks": table_fallbacks_present,
                            },
                            ensure_ascii=False,
                        )
                    ),
                }
            )

    return {
        "passed": all(bool(item["passed"]) for item in checklist),
        "elapsed_seconds": elapsed,
        "checks": checklist,
    }


if __name__ == "__main__":
    result = run_block_g_check(run_apptest="--skip-apptest" not in sys.argv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
