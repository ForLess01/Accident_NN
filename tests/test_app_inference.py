from __future__ import annotations

import inspect
import json
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.app_inference as inference
from src.app_inference import InputContractError, RuntimeArtifactError


@contextmanager
def raises(expected: type[BaseException], match: str):
    try:
        yield
    except expected as exc:
        assert re.search(match, str(exc)), f"{exc!r} does not match {match!r}"
    else:
        raise AssertionError(f"Expected {expected.__name__}")


def _canonical_record() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "FECHA": "2025-06-15", "HORA": "18:30", "DEPARTAMENTO": "LIMA",
            "CODIGO_VIA": "PE-1N", "LATITUD": -12.046374, "LONGITUD": -77.042793,
            "CLASE": "CHOQUE", "ZONA": "RURAL", "RED_VIAL": "NACIONAL",
            "TIPO_VIA": "CARRETERA", "CLIMA": "DESPEJADO",
            "CARACTERISTICA_VIA": "TRAMO RECTO", "PERFIL_VIA": "PLANA",
            "SUPERFICIE": "ASFALTADA",
            "n_vehiculos": 2, "n_bus": 1, "n_pesado_carga": 0, "n_moto": 0,
            "n_no_identificado": 0, "n_interprovincial": 1, "n_transporte_publico": 0,
            "n_personas": 4, "n_pasajeros": 2, "n_peatones": 0,
            "n_conductor_fugado": 0, "edad_media_involucrados": 35.0,
        }]
    )


def test_canonical_runtime_schema_and_outputs_align() -> None:
    manifest = inference.load_manifest()
    schema = inference.load_feature_schema()
    runtime = inference.load_prediction_stack()
    result = inference.predict_records(_canonical_record())

    assert manifest["feature_count"] == schema["processed_feature_count"] == 175
    assert int(runtime.model.input_shape[-1]) == 175
    assert runtime.encoders["feature_list"] == schema["processed_feature_order"]
    assert np.isfinite(result[["raw_probability", "calibrated_probability"]].to_numpy()).all()
    assert result["calibrated_probability"].between(0, 1).all()
    assert result["raw_probability"].between(0, 1).all()


def test_user_decision_uses_calibrated_scale_not_raw_scale() -> None:
    class FakeRuntime:
        calibration_method = "platt"

        def predict_dataframe(self, raw: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"raw_probability": [0.90], "calibrated_probability": [0.10], "raw_prediction": [1], "calibrated_prediction": [0]},
                index=raw.index,
            )

    thresholds = {
        "raw": {"value": 0.8, "probability_scale": "raw_mlp_sigmoid"},
        "calibrated": {"value": 0.3, "probability_scale": "platt_calibrated_probability"},
    }
    with (
        patch.object(inference, "load_prediction_stack", return_value=FakeRuntime()),
        patch.object(inference, "load_thresholds", return_value=thresholds),
    ):
        result = inference.predict_records(_canonical_record()).iloc[0]

    assert result["raw_prediction"] == 1
    assert result["priority_decision"] == "PRIORIDAD ESTÁNDAR"
    assert result["calibrated_threshold"] == 0.3
    assert result["raw_threshold"] == 0.8


def test_input_contract_rejects_partial_coordinates_and_missing_fields() -> None:
    partial = _canonical_record()
    partial.loc[0, "LONGITUD"] = None
    with raises(InputContractError, match="ambas coordenadas"):
        inference.predict_records(partial)

    absent = _canonical_record()
    absent.loc[0, ["LATITUD", "LONGITUD"]] = None
    with raises(InputContractError, match="obligatorias"):
        inference.predict_records(absent)

    missing = _canonical_record().drop(columns=["CLIMA"])
    with raises(InputContractError, match="CLIMA"):
        inference.predict_records(missing)


def test_input_contract_rejects_out_of_window_dates() -> None:
    for invalid_date in ("2020-12-31", "2026-01-01"):
        record = _canonical_record()
        record.loc[0, "FECHA"] = invalid_date
        with raises(InputContractError, match="01/01/2021.*31/12/2025"):
            inference.predict_records(record)


def test_peru_location_rejects_outside_and_department_mismatch() -> None:
    outside = _canonical_record()
    outside.loc[0, ["LATITUD", "LONGITUD"]] = [40.7128, -74.0060]
    with raises(InputContractError, match="fuera del territorio"):
        inference.predict_records(outside)

    mismatch = _canonical_record()
    mismatch.loc[0, "DEPARTAMENTO"] = "PUNO"
    with raises(InputContractError, match="no corresponden.*LIMA"):
        inference.predict_records(mismatch)


def test_geojson_point_in_polygon_supports_holes_multipolygons_and_boundaries() -> None:
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]
    assert inference.point_in_ring(5, 5, square) == 1
    assert inference.point_in_ring(0, 5, square) == 2
    assert inference.point_in_ring(11, 5, square) == 0
    assert inference.point_in_polygon(2, 2, [square, hole])
    assert not inference.point_in_polygon(5, 5, [square, hole])
    assert inference.point_in_polygon(3, 5, [square, hole])
    geometry = {"type": "MultiPolygon", "coordinates": [[square], [[[20, 20], [22, 20], [22, 22], [20, 22], [20, 20]]]]}
    assert inference.point_in_geometry(21, 21, geometry)
    assert not inference.point_in_geometry(15, 15, geometry)


def test_road_code_format_known_and_unseen_semantics() -> None:
    assert inference.is_known_road_code("PE-1N")
    assert not inference.is_known_road_code("PE-999X")
    unseen = _canonical_record()
    unseen.loc[0, "CODIGO_VIA"] = "pe-999x"
    result = inference.predict_records(unseen)
    assert np.isfinite(result["calibrated_probability"]).all()
    invalid = _canonical_record()
    invalid.loc[0, "CODIGO_VIA"] = "carretera inventada"
    with raises(InputContractError, match="formato como PE-1N"):
        inference.predict_records(invalid)


def test_missing_bundle_fails_controlled_and_does_not_create_files() -> None:
    inference.load_prediction_stack.cache_clear()
    with tempfile.TemporaryDirectory() as temporary_directory:
        empty = Path(temporary_directory) / "models" / "final"
        with patch.object(inference, "FINAL_MODEL_DIR", empty):
            with raises(RuntimeArtifactError, match="Falta un archivo"):
                inference.load_prediction_stack()
            assert not empty.exists()
    inference.load_prediction_stack.cache_clear()


def test_app_sources_are_read_only_and_use_canonical_evidence() -> None:
    inference_source = inspect.getsource(inference)
    app_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    forbidden = ("audit_and_clean", "to_parquet(", "joblib.dump")
    assert not any(token in inference_source for token in forbidden)
    assert "final_reference_metrics_2024_2025.json" in app_source
    assert "load_explainability_artifacts" in app_source
    assert "Todavía no se generó explicabilidad global" not in app_source
    assert "reference_artifact_hashes" in app_source
    assert "st.form(" in app_source
    assert "st.time_input(" in app_source
    assert "probability_gauge(calibrated_probability, calibrated_threshold)" in app_source
    assert 'st.session_state.pop("canonical_result", None)' in app_source
    assert 'st.query_params.get("section"' in app_source
    assert '"Cargar caso de demostración"' in app_source
    assert 'value=None' in app_source
    assert 'accept_new_options=True' in app_source
    assert '_table_fallback("matriz de confusión"' in app_source
    assert '_table_fallback("puntos de curvas PR y ROC"' in app_source
    assert "No usar para sancionar" not in app_source
    assert "no predice accidentes" not in app_source.lower()

    check_source = (ROOT / "src" / "block_g_app_check.py").read_text(encoding="utf-8")
    assert "run_apptest: bool = True" in check_source
    assert '"--skip-apptest" not in sys.argv' in check_source
    assert '"--apptest" in sys.argv' not in check_source


def test_manifest_threshold_scales_are_distinct() -> None:
    thresholds = json.loads((ROOT / "models" / "final" / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["raw"]["probability_scale"] != thresholds["calibrated"]["probability_scale"]
    assert thresholds["raw"]["value"] == 0.8
    assert thresholds["calibrated"]["value"] == 0.3


def test_schema_options_use_valid_raw_representatives() -> None:
    options = inference.load_input_options()
    assert "TRAMO RECTO" in options["CARACTERISTICA_VIA"]
    assert "RECTO" not in options["CARACTERISTICA_VIA"]
    assert "ASFALTADA" in options["SUPERFICIE"]
    assert "PAVIMENTADA" not in options["SUPERFICIE"]
    assert "VIA EXPRESA" in options["TIPO_VIA"]
    assert "EXPRESA" not in options["TIPO_VIA"]
    assert "DESCONOCIDO" not in options["CLASE"]


def test_final_schema_requires_latitude_and_runtime_requires_coordinate_pair() -> None:
    schema = inference.load_feature_schema()
    fields = {field["name"]: field for field in schema["required_raw_fields"]}
    for coordinate in ("LATITUD", "LONGITUD"):
        assert fields[coordinate]["required"] is True
        assert fields[coordinate]["nullable"] is False
    absent = _canonical_record()
    absent.loc[0, ["LATITUD", "LONGITUD"]] = None
    with raises(InputContractError, match="obligatorias"):
        inference.predict_records(absent)


def test_low_support_regional_rates_are_masked() -> None:
    regional = pd.DataFrame(
        [
            {"DEPARTAMENTO": "SOPORTADO", "siniestros_fatales": 30, "multifatales": 3, "tasa_multifatal": 0.10, "ci_95_inf": 0.03, "ci_95_sup": 0.25, "soporte_suficiente": True},
            {"DEPARTAMENTO": "INESTABLE", "siniestros_fatales": 29, "multifatales": 8, "tasa_multifatal": 8 / 29, "ci_95_inf": 0.14, "ci_95_sup": 0.46, "soporte_suficiente": True},
        ]
    )
    masked = inference.mask_unsupported_regional_rates(regional, minimum_support=30)
    supported = masked.loc[masked["DEPARTAMENTO"] == "SOPORTADO"].iloc[0]
    unstable = masked.loc[masked["DEPARTAMENTO"] == "INESTABLE"].iloc[0]
    assert supported["tasa_multifatal"] == 0.10
    assert bool(supported["soporte_suficiente"])
    assert pd.isna(unstable["tasa_multifatal"])
    assert pd.isna(unstable["ci_95_inf"])
    assert pd.isna(unstable["ci_95_sup"])
    assert not bool(unstable["soporte_suficiente"])


def test_matching_subgroup_is_masked_at_29_and_visible_at_30() -> None:
    comparison = pd.DataFrame(
        [
            {"comparador": "Subgrupo coincidente", "soporte": 29, "tasa_multifatal": 0.25, "ci_95_inf": 0.10, "ci_95_sup": 0.45},
            {"comparador": "Misma clase", "soporte": 29, "tasa_multifatal": 0.20, "ci_95_inf": 0.08, "ci_95_sup": 0.40},
        ]
    )
    masked = inference.mask_unsupported_historical_rates(comparison, minimum_support=30)
    subgroup = masked.loc[masked["comparador"] == "Subgrupo coincidente"].iloc[0]
    same_class = masked.loc[masked["comparador"] == "Misma clase"].iloc[0]
    assert pd.isna(subgroup["tasa_multifatal"])
    assert not bool(subgroup["soporte_suficiente"])
    assert same_class["tasa_multifatal"] == 0.20

    comparison.loc[0, "soporte"] = 30
    visible = inference.mask_unsupported_historical_rates(comparison, minimum_support=30).iloc[0]
    assert visible["tasa_multifatal"] == 0.25
    assert bool(visible["soporte_suficiente"])


if __name__ == "__main__":
    test_canonical_runtime_schema_and_outputs_align()
    test_user_decision_uses_calibrated_scale_not_raw_scale()
    test_input_contract_rejects_partial_coordinates_and_missing_fields()
    test_input_contract_rejects_out_of_window_dates()
    test_peru_location_rejects_outside_and_department_mismatch()
    test_geojson_point_in_polygon_supports_holes_multipolygons_and_boundaries()
    test_road_code_format_known_and_unseen_semantics()
    test_missing_bundle_fails_controlled_and_does_not_create_files()
    test_app_sources_are_read_only_and_use_canonical_evidence()
    test_manifest_threshold_scales_are_distinct()
    test_schema_options_use_valid_raw_representatives()
    test_final_schema_requires_latitude_and_runtime_requires_coordinate_pair()
    test_low_support_regional_rates_are_masked()
    test_matching_subgroup_is_masked_at_29_and_visible_at_30()
    print("app-inference-ok")
