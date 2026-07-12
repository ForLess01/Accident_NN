from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_protocol import (
    EXCLUDED_COLUMNS,
    derive_base_features,
    fit_preprocessor,
    split_chronological,
    transform_features,
)
import src.block_e_modeling as modeling


def _row(date: str, target: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "FECHA": date,
        "HORA": "23:30",
        "DEPARTAMENTO": "PUNO",
        "CODIGO_VIA": "PE-3S",
        "LATITUD": -15.84,
        "LONGITUD": -70.02,
        "CLASE": "CHOQUE",
        "ZONA": "RURAL",
        "RED_VIAL": "NACIONAL",
        "TIPO_VIA": "CARRETERA",
        "CLIMA": "LLUVIOSO",
        "CARACTERISTICA_VIA": "CURVA",
        "PERFIL_VIA": "INCLINADA",
        "SUPERFICIE": "ASFALTADA",
        "target_multifatal": target,
        "FALLECIDOS": target + 1,
        "LESIONADOS": 3,
        "CAUSA_FACTOR": "POSTERIOR",
        "CODIGO_SINIESTRO": f"id-{date}",
    }
    row.update(overrides)
    return row


def test_split_is_chronological_and_excludes_leakage() -> None:
    base = pd.DataFrame([
        _row("2021-01-01", 0), _row("2022-12-31", 1),
        _row("2023-06-01", 0), _row("2024-01-01", 1), _row("2025-01-01", 0),
    ])
    splits = split_chronological(base)
    assert pd.to_datetime(splits["X_train_raw"]["FECHA"]).dt.year.max() == 2022
    assert pd.to_datetime(splits["X_validation_raw"]["FECHA"]).dt.year.unique().tolist() == [2023]
    assert pd.to_datetime(splits["X_test_raw"]["FECHA"]).dt.year.min() == 2024
    for name in ("X_train_raw", "X_validation_raw", "X_test_raw"):
        assert not (set(splits[name].columns) & EXCLUDED_COLUMNS)


def test_feature_contract_has_cyclic_and_predeclared_interactions() -> None:
    train = pd.DataFrame([_row("2021-01-01", 0), _row("2022-07-01", 1)])
    sample = pd.DataFrame([_row("2023-12-31", 1, CODIGO_VIA="PE-999X")])
    scaler, encoders = fit_preprocessor(train)
    transformed = transform_features(sample, scaler, encoders)
    derived = derive_base_features(sample, encoders["via_frequency_map"])
    assert transformed.columns.tolist() == encoders["feature_list"]
    assert transformed.isna().sum().sum() == 0
    assert {"mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos", "night_rural", "rain_curve"}.issubset(transformed.columns)
    assert derived.loc[0, "night_rural"] == 1
    assert derived.loc[0, "rain_curve"] == 1
    assert transformed.filter(like="road_type_zone_carretera__rural").iloc[0, 0] == 1
    assert transformed.filter(like="road_network_class_nacional__choque").iloc[0, 0] == 1
    assert "anio" not in transformed.columns


def test_modeling_pipeline_does_not_transform_reference_period() -> None:
    train_raw = pd.DataFrame([_row("2021-01-01", 0), _row("2022-02-01", 1)])
    selection_raw = pd.DataFrame([_row("2023-01-01", 0), _row("2023-02-01", 1)])
    reference_raw = pd.DataFrame([_row("2024-01-01", 0), _row("2025-01-01", 1)])
    y_train = pd.Series([0, 1], dtype="int8")
    y_selection = pd.Series([0, 1], dtype="int8")
    splits = {
        "X_train_raw": train_raw,
        "y_train": y_train,
        "X_validation_raw": selection_raw,
        "y_validation": y_selection,
        "X_test_raw": reference_raw,
        "y_test": pd.Series([0, 1], dtype="int8"),
    }
    transformed: list[pd.DataFrame] = []

    class FakeModel:
        def save(self, path: Path) -> None:
            Path(path).write_text("model", encoding="utf-8")

    config = modeling.MLPConfig("MLP_64_32", (64, 32), 0.35, 3e-4, 5e-4, 64)
    threshold = {"threshold": 0.65, "selection_policy": "validation only"}
    grid = pd.DataFrame({"config_id": ["MLP_64_32"], "seed": [314]})
    robustness = pd.DataFrame({"config_id": ["MLP_64_32"], "median_val_pr_auc": [0.25]})
    baselines = pd.DataFrame({"model": ["LogisticRegression_balanced"], "threshold": [0.6]})

    def fake_transform(raw: pd.DataFrame, scaler: object, encoders: dict[str, object]) -> pd.DataFrame:
        assert raw is not reference_raw
        transformed.append(raw)
        return pd.DataFrame({"feature": [0.0, 1.0]})

    with tempfile.TemporaryDirectory() as temporary_directory:
        temp = Path(temporary_directory)
        with (
            patch.object(modeling.pd, "read_parquet", return_value=pd.DataFrame()),
            patch.object(modeling, "split_chronological", return_value=splits),
            patch.object(modeling, "fit_preprocessor", return_value=(object(), {"feature_list": ["feature"]})),
            patch.object(modeling, "transform_features", side_effect=fake_transform),
            patch.object(modeling, "train_baselines", return_value=({}, baselines)),
            patch.object(modeling, "train_mlp_grid", return_value=(FakeModel(), config, 314, threshold, np.array([0.2, 0.8]), grid, robustness)),
            patch.object(modeling.joblib, "dump", side_effect=lambda value, path: Path(path).write_text("artifact", encoding="utf-8")),
            patch.object(modeling, "feature_availability_audit", return_value=pd.DataFrame({"feature": ["x"]})),
            patch.object(modeling, "FINAL_MODEL_DIR", temp / "models"),
            patch.object(modeling, "TABLES_DIR", temp / "tables"),
        ):
            summary = modeling.run_modeling_pipeline()
    assert transformed == [train_raw, selection_raw]
    assert summary["reference_period_used_for_selection"] is False


def test_baseline_helper_selects_each_threshold_from_validation_labels() -> None:
    """Protect the helper itself, not only the orchestration that calls it."""
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 1, 1], dtype="int8")
    X_validation = pd.DataFrame({"feature": [4.0, 5.0]})
    y_validation = pd.Series([0, 1], dtype="int8")
    endpoint_label_sentinel = pd.Series([1, 0], dtype="int8")
    selected_with: list[pd.Series] = []
    original_choose_threshold = modeling.choose_threshold

    class FakeBaseline:
        classes_ = np.array([0, 1])

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> "FakeBaseline":
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    def guarded_choose(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
        selected_with.append(labels)
        assert labels is y_validation
        return original_choose_threshold(labels, probabilities)

    with (
        patch.object(modeling, "LogisticRegression", lambda **_: FakeBaseline()),
        patch.object(modeling, "RandomForestClassifier", lambda **_: FakeBaseline()),
        patch.object(modeling, "choose_threshold", side_effect=guarded_choose),
        # If a future mutation literally substitutes y_val with y_test inside
        # the helper, it resolves to this distinct sentinel and fails identity.
        patch.object(modeling, "y_test", endpoint_label_sentinel, create=True),
    ):
        _, table = modeling.train_baselines(X_train, y_train, X_validation, y_validation)

    assert len(selected_with) == 3
    assert table["threshold"].notna().all()


def test_mlp_grid_helper_selects_every_run_threshold_from_validation_labels() -> None:
    """A future refactor cannot silently substitute endpoint labels inside the grid."""
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 1, 1], dtype="int8")
    X_validation = pd.DataFrame({"feature": [4.0, 5.0]})
    y_validation = pd.Series([0, 1], dtype="int8")
    endpoint_label_sentinel = pd.Series([1, 0], dtype="int8")
    selected_with: list[pd.Series] = []
    original_choose_threshold = modeling.choose_threshold

    class FakeHistory:
        history = {"loss": [0.7], "val_pr_auc": [0.6]}

    class FakeMlp:
        def fit(self, *args: object, **kwargs: object) -> FakeHistory:
            return FakeHistory()

        def predict(self, features: pd.DataFrame, verbose: int = 0) -> np.ndarray:
            return np.array([[0.2], [0.8]])

    def guarded_choose(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
        selected_with.append(labels)
        assert labels is y_validation
        return original_choose_threshold(labels, probabilities)

    with (
        patch.object(modeling, "set_global_seed", return_value=None),
        patch.object(modeling, "build_mlp", side_effect=lambda *args: FakeMlp()),
        patch.object(modeling, "choose_threshold", side_effect=guarded_choose),
        # Deliberately define a distinct endpoint label so y_val -> y_test
        # mutations are caught rather than raising NameError in the test.
        patch.object(modeling, "y_test", endpoint_label_sentinel, create=True),
    ):
        _, _, _, _, _, grid, _ = modeling.train_mlp_grid(X_train, y_train, X_validation, y_validation)

    assert len(grid) == len(modeling.MLP_GRID) * len(modeling.SEEDS)
    # One call per grid run plus one call for the selected seed's frozen threshold.
    assert len(selected_with) == len(grid) + 1


if __name__ == "__main__":
    test_split_is_chronological_and_excludes_leakage()
    test_feature_contract_has_cyclic_and_predeclared_interactions()
    test_modeling_pipeline_does_not_transform_reference_period()
    test_baseline_helper_selects_each_threshold_from_validation_labels()
    test_mlp_grid_helper_selects_every_run_threshold_from_validation_labels()
    print("model-protocol-ok")
