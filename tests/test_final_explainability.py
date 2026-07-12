from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.app_inference as app_inference
from src.final_explainability import (
    EXPLANATION_RAW_COLUMNS,
    aggregate_global_explanations,
    load_pre_endpoint_source,
    partition_explainability_source,
    sha256_file,
)


def test_source_boundary_loads_no_endpoint_rows_or_labels() -> None:
    frame = load_pre_endpoint_source(ROOT / "data" / "processed" / "base_limpia.parquet")
    background, explained = partition_explainability_source(frame)
    assert frame.columns.tolist() == EXPLANATION_RAW_COLUMNS
    assert pd.to_datetime(frame["FECHA"]).max() < pd.Timestamp("2024-01-01")
    assert set(pd.to_datetime(background["FECHA"]).dt.year.unique()) == {2021, 2022}
    assert set(pd.to_datetime(explained["FECHA"]).dt.year.unique()) == {2023}
    assert not {"target_multifatal", "FALLECIDOS", "LESIONADOS", "VEHICULOS_DANADOS"}.intersection(frame.columns)


def test_source_reader_enforces_projection_and_pre_2024_predicate_on_every_read() -> None:
    source_path = ROOT / "data" / "processed" / "base_limpia.parquet"
    safe_frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2021-01-01", "2023-12-31"])
                if column == "FECHA"
                else ["00:00", "12:00"]
                if column == "HORA"
                else [0, 12]
                if column == "hora_entera"
                else [0.0, 0.0]
                if column in {"LATITUD", "LONGITUD"}
                else ["SAFE", "SAFE"]
            )
            for column in EXPLANATION_RAW_COLUMNS
        }
    )
    observed_reads = 0

    def strict_reader(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal observed_reads
        observed_reads += 1
        assert Path(path) == source_path
        assert not args, "Unsafe positional read_parquet arguments are forbidden."
        assert set(kwargs) == {"columns", "filters"}, "Every read must declare only the safe projection and predicate."
        assert kwargs["columns"] == EXPLANATION_RAW_COLUMNS
        assert kwargs["filters"] == [("FECHA", "<", pd.Timestamp("2024-01-01"))]
        return safe_frame.copy()

    with patch("src.final_explainability.pd.read_parquet", side_effect=strict_reader) as reader:
        loaded = load_pre_endpoint_source(source_path)

    assert observed_reads == reader.call_count == 1
    assert loaded.columns.tolist() == EXPLANATION_RAW_COLUMNS
    assert pd.to_datetime(loaded["FECHA"]).max() < pd.Timestamp("2024-01-01")


def test_group_aggregation_is_finite_signed_and_additive() -> None:
    features = ["mes_sin", "mes_cos", "clima_DESPEJADO", "rain_curve"]
    values = np.array([[0.2, -0.1, 0.3, -0.2], [-0.4, 0.2, -0.1, 0.5]], dtype=float)
    groups, feature_table = aggregate_global_explanations(values, features)
    assert np.isfinite(groups.select_dtypes(include="number").to_numpy()).all()
    assert np.isfinite(feature_table.select_dtypes(include="number").to_numpy()).all()
    fecha = groups.set_index("raw_variable_group").loc["FECHA"]
    expected = values[:, :2].sum(axis=1)
    assert np.isclose(fecha["mean_abs_grouped_shap"], np.mean(np.abs(expected)))
    assert np.isclose(fecha["mean_signed_grouped_shap"], np.mean(expected))


def test_persisted_explainability_schema_values_and_hashes() -> None:
    manifest = json.loads((ROOT / "models" / "final" / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "report" / "tables" / "final_explainability_provenance.json").read_text(encoding="utf-8"))
    groups = pd.read_csv(ROOT / "report" / "tables" / "final_explainability_group_importance.csv")
    features = pd.read_csv(ROOT / "report" / "tables" / "final_explainability_feature_importance.csv")
    required_group_columns = {
        "rank", "raw_variable_group", "processed_feature_count", "mean_abs_grouped_shap",
        "mean_signed_grouped_shap", "positive_contribution_share", "average_direction", "importance_share",
    }
    assert required_group_columns.issubset(groups.columns)
    assert len(features) == manifest["feature_count"] == 162
    assert groups["processed_feature_count"].sum() == 162
    assert np.isfinite(groups.select_dtypes(include="number").to_numpy()).all()
    assert np.isfinite(features.select_dtypes(include="number").to_numpy()).all()
    assert provenance["background_partition"] == "training_2021_2022_only"
    assert provenance["explanation_partition"] == "validation_2023_only"
    assert provenance["labels_loaded"] is False
    assert provenance["endpoint_2024_2025_data_loaded"] is False
    assert provenance["model_selection_performed"] is False
    assert provenance["weights_or_calibration_modified"] is False
    assert provenance["association_not_causality"] is True
    assert provenance["local_explanations_generated"] is False
    for relative_path, expected in manifest["explainability_artifact_hashes"].items():
        assert sha256_file(ROOT / relative_path) == expected


def test_ui_loader_returns_hash_verified_validation_only_evidence() -> None:
    app_inference.load_manifest.cache_clear()
    app_inference.load_explainability_artifacts.cache_clear()
    evidence = app_inference.load_explainability_artifacts()
    assert not evidence["groups"].empty
    assert len(evidence["features"]) == 162
    assert evidence["provenance"]["explanation_partition"] == "validation_2023_only"
    assert evidence["figure"].name == "final_explainability_global.png"


if __name__ == "__main__":
    test_source_boundary_loads_no_endpoint_rows_or_labels()
    test_source_reader_enforces_projection_and_pre_2024_predicate_on_every_read()
    test_group_aggregation_is_finite_signed_and_additive()
    test_persisted_explainability_schema_values_and_hashes()
    test_ui_loader_returns_hash_verified_validation_only_evidence()
    print("final-explainability-ok")
