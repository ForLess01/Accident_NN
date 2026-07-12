from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.final_model_bundle as final_bundle
from src.final_model_bundle import (
    CanonicalModelBundle,
    select_calibrator_validation_only,
    sha256_file,
)
from src.model_protocol import EXCLUDED_COLUMNS, split_chronological


FINAL_DIR = ROOT / "models" / "final"
REFERENCE_PROBABILITIES = ROOT / "report" / "tables" / "final_reference_probabilities_2024_2025.csv"


def test_manifest_schema_artifacts_and_hashes_are_valid() -> None:
    manifest = json.loads((FINAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "manifest_schema_version",
        "model_version",
        "dataset",
        "code_and_libraries",
        "splits",
        "feature_count",
        "architecture",
        "calibration",
        "thresholds",
        "artifact_hashes",
        "reference_evaluation",
    }
    assert required.issubset(manifest)
    assert manifest["model_version"] == "canonical-1.0.0"
    assert manifest["feature_count"] == 162
    assert manifest["architecture"]["weights_frozen"] is True
    assert manifest["architecture"]["architecture_search_rerun"] is False
    assert manifest["reference_evaluation"]["used_for_model_selection"] is False
    assert manifest["reference_evaluation"]["used_for_calibration_selection"] is False
    for name, expected_hash in manifest["artifact_hashes"].items():
        assert sha256_file(FINAL_DIR / name) == expected_hash
    for name, expected_hash in manifest["reference_artifact_hashes"].items():
        assert sha256_file(ROOT / "report" / "tables" / name) == expected_hash
    for relative_path, expected_hash in manifest["evaluation_figure_hashes"].items():
        assert sha256_file(ROOT / relative_path) == expected_hash
    assert manifest["evaluation_figure_generator_sha256"] == sha256_file(
        ROOT / "src" / "final_evaluation_figures.py"
    )
    selection_paths = {
        "model_selection.json": FINAL_DIR / "model_selection.json",
        "model_selection_baseline_validation.csv": ROOT / "report" / "tables" / "model_selection_baseline_validation.csv",
        "model_selection_seed_grid_validation.csv": ROOT / "report" / "tables" / "model_selection_seed_grid_validation.csv",
        "model_selection_robustness.csv": ROOT / "report" / "tables" / "model_selection_robustness.csv",
    }
    for name, expected_hash in manifest["selection_artifact_hashes"].items():
        assert sha256_file(selection_paths[name]) == expected_hash
    assert manifest["code_and_libraries"]["builder_code_sha256"] == sha256_file(
        ROOT / "src" / "final_model_bundle.py"
    )
    assert manifest["code_and_libraries"]["feature_protocol_sha256"] == sha256_file(
        ROOT / "src" / "model_protocol.py"
    )
    assert sha256_file(ROOT / manifest["dataset"]["path"]) == manifest["dataset"]["sha256"]


def test_persisted_calibrated_threshold_uses_oof_validation_semantics() -> None:
    thresholds = json.loads((FINAL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    evidence = json.loads((FINAL_DIR / "calibration_selection.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(REFERENCE_PROBABILITIES)
    calibrated_threshold = float(thresholds["calibrated"]["value"])
    assert thresholds["calibrated"]["source_partition"] == "validation_2023_oof"
    assert calibrated_threshold == float(evidence["selected_calibrated_threshold"]["threshold"])
    assert evidence["threshold_source"] == "selected method OOF calibrated validation predictions"
    assert np.array_equal(
        reference["calibrated_prediction"].to_numpy(),
        (reference["calibrated_probability"].to_numpy() >= calibrated_threshold).astype(int),
    )


def test_calibration_selector_accepts_validation_only_and_never_endpoint_labels() -> None:
    signature = inspect.signature(select_calibrator_validation_only)
    assert "test_labels" not in signature.parameters
    validation_probabilities = np.linspace(0.01, 0.99, 200)
    validation_labels = pd.Series(([0] * 160) + ([1] * 40), dtype="int8")
    endpoint_label_sentinel = np.ones(17, dtype=int)
    observed_fit_labels: list[np.ndarray] = []
    original_fit = final_bundle._fit_calibrator

    def guarded_fit(method: str, probabilities: np.ndarray, labels: np.ndarray) -> object:
        observed = np.asarray(labels).copy()
        observed_fit_labels.append(observed)
        assert not np.array_equal(observed, endpoint_label_sentinel)
        assert len(observed) in {160, 200}
        return original_fit(method, probabilities, labels)

    with (
        patch.object(final_bundle, "_fit_calibrator", side_effect=guarded_fit),
        patch.object(final_bundle, "y_test", endpoint_label_sentinel, create=True),
    ):
        method, _, evidence, oof = select_calibrator_validation_only(
            validation_probabilities, validation_labels
        )

    assert method in {"platt", "isotonic"}
    assert np.isfinite(oof).all()
    assert evidence["source_partition"] == "validation_2023_only"
    assert evidence["test_labels_used_for_selection"] is False
    assert len(observed_fit_labels) == (2 * final_bundle.CALIBRATION_FOLDS) + 1


def test_bundle_builder_freezes_validation_selection_before_endpoint_open() -> None:
    """Exercise the builder boundary, not only the calibration helper."""
    train_raw = pd.DataFrame({"FECHA": ["2021-01-01", "2022-01-01"], "DEPARTAMENTO": ["PUNO", "LIMA"]})
    validation_raw = pd.DataFrame({"FECHA": ["2023-01-01", "2023-02-01"], "DEPARTAMENTO": ["PUNO", "LIMA"]})
    endpoint_raw = pd.DataFrame({"FECHA": ["2024-01-01", "2025-01-01"], "DEPARTAMENTO": ["PUNO", "LIMA"]})
    y_train = pd.Series([0, 1], dtype="int8")
    y_validation = pd.Series([0, 1], dtype="int8")
    endpoint_label_sentinel = pd.Series([1, 0], dtype="int8")
    splits = {
        "X_train_raw": train_raw,
        "y_train": y_train,
        "X_validation_raw": validation_raw,
        "y_validation": y_validation,
        "X_test_raw": endpoint_raw,
        "y_test": endpoint_label_sentinel,
    }
    endpoint_opened = False
    selection_calls: list[pd.Series] = []

    class FakeModel:
        input_shape = (None, 1)

        def predict(self, features: pd.DataFrame, verbose: int = 0) -> np.ndarray:
            return np.array([[0.2], [0.8]], dtype=float)

    def guarded_transform(raw: pd.DataFrame, scaler: object, encoders: dict[str, object]) -> pd.DataFrame:
        if raw is endpoint_raw:
            assert endpoint_opened, "Endpoint features must not be transformed before selection is frozen."
        return pd.DataFrame({"feature": [0.0, 1.0]})

    def guarded_selector(probabilities: np.ndarray, labels: pd.Series) -> tuple[str, object, dict[str, object], np.ndarray]:
        assert not endpoint_opened
        assert labels is y_validation
        assert labels is not endpoint_label_sentinel
        selection_calls.append(labels)
        evidence: dict[str, object] = {
            "source_partition": "validation_2023_only",
            "test_labels_used_for_selection": False,
            "selected_method": "platt",
        }
        return "platt", object(), evidence, np.array([0.1, 0.9])

    def guarded_threshold(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
        assert not endpoint_opened
        assert labels is y_validation
        assert labels is not endpoint_label_sentinel
        selection_calls.append(labels)
        return {"threshold": 0.5, "selection_policy": "validation-only test policy"}

    def open_endpoint() -> None:
        nonlocal endpoint_opened
        assert len(selection_calls) == 2
        endpoint_opened = True

    minimal_metrics = {
        "n": 2,
        "positives": 1,
        "class_rate": 0.5,
        "threshold": 0.5,
        "f1_multifatal": 1.0,
        "precision_multifatal": 1.0,
        "recall_multifatal": 1.0,
        "pr_auc": 1.0,
        "roc_auc": 1.0,
        "brier": 0.1,
        "ece_10_bins": 0.1,
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        final_dir = root / "models" / "final"
        tables_dir = root / "report" / "tables"
        processed_dir = root / "data" / "processed"
        final_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        (root / "src").mkdir()
        (root / "src" / "model_protocol.py").write_text("# fake protocol\n", encoding="utf-8")
        (processed_dir / "base_limpia.parquet").write_bytes(b"fake-dataset")
        (final_dir / "model.keras").write_bytes(b"fake-model")
        (final_dir / "scaler.joblib").write_bytes(b"fake-scaler")
        (final_dir / "encoders.joblib").write_bytes(b"fake-encoders")
        (final_dir / "feature_list.json").write_text('["feature"]', encoding="utf-8")
        (final_dir / "model_selection.json").write_text(
            json.dumps(
                {
                    "selected_config": {
                        "config_id": "MLP_64_32",
                        "hidden_units": [64, 32],
                        "dropout": 0.35,
                        "l2": 0.0003,
                        "learning_rate": 0.0005,
                        "batch_size": 64,
                    },
                    "selected_seed": 314,
                    "selected_threshold": {"threshold": 0.65},
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "model_selection_baseline_validation.csv",
            "model_selection_seed_grid_validation.csv",
            "model_selection_robustness.csv",
        ):
            (tables_dir / name).write_text("placeholder\n", encoding="utf-8")
        fake_base = pd.DataFrame({"placeholder": [1, 2]})
        fake_confusion = pd.DataFrame({"probability_scale": ["raw"], "actual": [0], "predicted": [0], "count": [1]})
        fake_report = pd.DataFrame({"probability_scale": ["raw"], "class_or_average": ["multifatal"]})
        fake_ci = pd.DataFrame(
            {
                "probability_scale": ["raw"],
                "metric": ["f1_multifatal"],
                "estimate": [1.0],
                "ci_2_5": [1.0],
                "ci_97_5": [1.0],
                "bootstrap_iterations": [1],
            }
        )
        fake_baseline = pd.DataFrame(
            [{"model": "baseline", "probability_scale": "raw", "threshold_source": "validation", **minimal_metrics}]
        )

        with (
            patch.object(final_bundle.pd, "read_parquet", return_value=fake_base),
            patch.object(final_bundle, "split_chronological", return_value=splits),
            patch.object(final_bundle.joblib, "load", side_effect=[object(), {"feature_list": ["feature"]}]),
            patch.object(final_bundle.keras.models, "load_model", return_value=FakeModel()),
            patch.object(final_bundle, "transform_features", side_effect=guarded_transform),
            patch.object(final_bundle, "select_calibrator_validation_only", side_effect=guarded_selector),
            patch.object(final_bundle, "choose_threshold", side_effect=guarded_threshold),
            patch.object(final_bundle, "apply_calibrator", return_value=np.array([0.1, 0.9])),
            patch.object(final_bundle.joblib, "dump", side_effect=lambda value, path: Path(path).write_bytes(b"fake-calibrator")),
            patch.object(final_bundle, "_input_schema", return_value={"processed_feature_count": 1}),
            patch.object(final_bundle, "_metrics", return_value=minimal_metrics),
            patch.object(final_bundle, "_bootstrap_metrics", return_value=fake_ci),
            patch.object(final_bundle, "_classification_tables", return_value=(fake_confusion, fake_report)),
            patch.object(final_bundle, "_baseline_reference", return_value=fake_baseline),
            # If production selection is literally mutated from y_val to y_test,
            # this global resolves to the distinct endpoint sentinel and the
            # identity assertions above fail semantically.
            patch.object(final_bundle, "y_test", endpoint_label_sentinel, create=True),
        ):
            manifest = final_bundle.build_canonical_bundle(
                root=root, bootstrap_iterations=1, endpoint_opened_hook=open_endpoint
            )

    assert endpoint_opened
    assert len(selection_calls) == 2 and all(labels is y_validation for labels in selection_calls)
    assert manifest["reference_evaluation"]["used_for_calibration_selection"] is False


def test_persisted_inference_matches_reference_and_outputs_are_finite() -> None:
    base = pd.read_parquet(ROOT / "data" / "processed" / "base_limpia.parquet")
    raw_test = split_chronological(base)["X_test_raw"]
    runtime = CanonicalModelBundle(FINAL_DIR)
    predictions = runtime.predict_dataframe(raw_test)  # type: ignore[arg-type]
    reference = pd.read_csv(REFERENCE_PROBABILITIES)
    assert len(predictions) == len(reference) == 2232
    assert np.isfinite(predictions.to_numpy()).all()
    assert predictions[["raw_probability", "calibrated_probability"]].min().min() >= 0
    assert predictions[["raw_probability", "calibrated_probability"]].max().max() <= 1
    np.testing.assert_allclose(
        predictions["raw_probability"].to_numpy(), reference["raw_probability"].to_numpy(), rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        predictions["calibrated_probability"].to_numpy(),
        reference["calibrated_probability"].to_numpy(),
        rtol=0,
        atol=1e-7,
    )
    assert np.array_equal(predictions["raw_prediction"].to_numpy(), reference["raw_prediction"].to_numpy())
    assert np.array_equal(
        predictions["calibrated_prediction"].to_numpy(), reference["calibrated_prediction"].to_numpy()
    )


def test_final_feature_contract_excludes_endpoint_and_post_event_leakage() -> None:
    schema = json.loads((FINAL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    features = set(json.loads((FINAL_DIR / "feature_list.json").read_text(encoding="utf-8")))
    raw_fields = {field["name"]: field for field in schema["required_raw_fields"]}
    assert schema["schema_version"] == "1.1"
    for coordinate in ("LATITUD", "LONGITUD"):
        assert raw_fields[coordinate]["required"] is True
        assert raw_fields[coordinate]["nullable"] is False
    assert schema["runtime_constraints"]["coordinate_pair"] == {
        "fields": ["LATITUD", "LONGITUD"],
        "required": True,
        "nullable": False,
    }
    assert schema["processed_feature_count"] == len(features) == 162
    assert not (features & set(EXCLUDED_COLUMNS))
    assert {"FALLECIDOS", "LESIONADOS", "CAUSA_FACTOR", "CAUSA_ESPECIFICA"}.issubset(
        set(schema["excluded_leakage_columns"])
    )


if __name__ == "__main__":
    test_manifest_schema_artifacts_and_hashes_are_valid()
    test_persisted_calibrated_threshold_uses_oof_validation_semantics()
    test_calibration_selector_accepts_validation_only_and_never_endpoint_labels()
    test_bundle_builder_freezes_validation_selection_before_endpoint_open()
    test_persisted_inference_matches_reference_and_outputs_are_finite()
    test_final_feature_contract_excludes_endpoint_and_post_event_leakage()
    print("final-model-bundle-ok")
