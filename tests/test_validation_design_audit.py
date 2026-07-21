from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.validation_design_audit as design_audit
from src.final_model_bundle import DESIGN_EVIDENCE_PATHS, design_evidence_hashes, sha256_file

TABLES = ROOT / "report" / "tables"


def test_design_audit_uses_validation_only_for_decisions() -> None:
    audit = json.loads((TABLES / "design_validation_audit.json").read_text(encoding="utf-8"))
    assert audit["selection_boundary"]["fit"] == "2021-2022"
    assert audit["selection_boundary"]["design_evaluation"] == "2023"
    assert audit["reference_is_post_hoc"] is True
    assert audit["canonical_artifacts_modified"] is False
    assert audit["raw_input_fields"] == 26
    assert audit["processed_features"] == 175


def test_endpoint_read_is_structurally_guarded_by_frozen_decisions() -> None:
    run_source = inspect.getsource(design_audit.run)
    assert run_source.index("_freeze_design_decisions") < run_source.index("_read_reference_after_freeze")
    reference = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "target_multifatal": [0, 1],
            "n_personas": [2, 5],
        }
    )
    probabilities = pd.DataFrame({"raw_probability": [0.1, 0.9]})
    with (
        patch.object(design_audit.pd, "read_parquet", return_value=reference) as parquet_read,
        patch.object(design_audit.pd, "read_csv", return_value=probabilities),
    ):
        try:
            design_audit._read_reference_after_freeze(None)  # type: ignore[arg-type]
        except RuntimeError as exc:
            assert "requires frozen" in str(exc)
        else:
            raise AssertionError("Reference access accepted a missing freeze guard.")
        parquet_read.assert_not_called()
        frozen = design_audit.FrozenDesignDecisions(4.0, "retain_single_seed314_frozen", True)
        X_reference, y_reference, loaded_probabilities = design_audit._read_reference_after_freeze(frozen)
        assert len(X_reference) == len(y_reference) == len(loaded_probabilities) == 2
        assert parquet_read.call_args.kwargs["filters"] == [("FECHA", ">=", design_audit.datetime(2024, 1, 1))]


def test_design_evidence_hash_builder_is_complete_and_stably_ordered() -> None:
    hashes = design_evidence_hashes(ROOT)
    expected_keys = {
        str(path) if path.parts[1] == "figures" else path.name
        for path in DESIGN_EVIDENCE_PATHS
    }
    assert set(hashes) == expected_keys
    assert list(hashes) == sorted(hashes)


def test_one_network_conclusion_follows_paired_intervals() -> None:
    strategies = pd.read_csv(TABLES / "design_network_strategy_validation.csv")
    paired = pd.read_csv(TABLES / "design_network_strategy_bootstrap.csv")
    assert set(strategies["strategy"]) == {
        "single_seed314_frozen",
        "ensemble_mean_3_seeds",
        "multibranch_162_context_13_companion_mean_3_seeds",
    }
    assert ((paired["ci_2_5"] <= 0) & (paired["ci_97_5"] >= 0)).all()
    assert set(paired["partition"]) == {"validation_2023"}


def test_person_baseline_threshold_is_validation_selected_and_scales_are_labeled() -> None:
    comparison = pd.read_csv(TABLES / "design_n_personas_reference_comparison.csv")
    paired = pd.read_csv(TABLES / "design_n_personas_paired_bootstrap.csv")
    baseline = comparison.loc[comparison["model"].eq("regla_n_personas")].iloc[0]
    mlp = comparison.loc[comparison["model"].eq("MLP_canónica_cruda")].iloc[0]
    assert baseline["selection_partition"] == "validation_2023_only"
    assert baseline["probability_scale"] == "ordinal_count_not_probability"
    assert baseline["threshold"] == 4
    assert mlp["probability_scale"] == "raw_mlp_sigmoid"
    assert (paired["partition"] == "historical_reference_2024_2025_post_hoc").all()


def test_design_evidence_and_training_code_are_hashed() -> None:
    manifest = json.loads((ROOT / "models" / "final" / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["design_evidence_artifact_hashes"].items():
        path = ROOT / name if "/" in name else TABLES / name
        assert sha256_file(path) == expected
    assert manifest["code_and_libraries"]["training_code_sha256"] == sha256_file(ROOT / "src" / "block_e_modeling.py")
    assert manifest["code_and_libraries"]["design_audit_code_sha256"] == sha256_file(ROOT / "src" / "validation_design_audit.py")


def test_visible_claims_match_canonical_artifacts_and_avoid_overclaiming() -> None:
    manifest = json.loads((ROOT / "models" / "final" / "manifest.json").read_text(encoding="utf-8"))
    raw_ci = pd.read_csv(TABLES / "final_reference_bootstrap_ci_2024_2025.csv")
    raw_ci = raw_ci.loc[raw_ci["probability_scale"].eq("raw")].set_index("metric")
    report = (ROOT / "report" / "sections" / "10_resultados.tex").read_text(encoding="utf-8")
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", ROOT / "Plan.md", ROOT / "docs" / "defensa_10min.md", *sorted((ROOT / "report" / "sections").glob("*.tex"))]
    ).lower()
    assert 'metrics["roc_auc"]' in app
    assert "ROC-AUC 0,75" not in app
    assert "0.4957 [0.4379, 0.5462]" in report
    assert round(raw_ci.loc["f1_multifatal", "ci_2_5"], 4) == 0.4379
    assert round(raw_ci.loc["precision_multifatal", "ci_2_5"], 4) == 0.4078
    assert round(raw_ci.loc["recall_multifatal", "ci_97_5"], 4) == 0.5850
    assert "un tercio" not in corpus
    assert "empate estadístico" not in corpus
    assert "techo informacional, demostrado" not in corpus
    assert "disponible al caracterizar la notificación" not in corpus
    assert "no participó en ninguna decisión" not in corpus
    assert "tres arquitecturas" not in corpus
    assert "disponible al registrar inicialmente" not in corpus


if __name__ == "__main__":
    test_design_audit_uses_validation_only_for_decisions()
    test_endpoint_read_is_structurally_guarded_by_frozen_decisions()
    test_design_evidence_hash_builder_is_complete_and_stably_ordered()
    test_one_network_conclusion_follows_paired_intervals()
    test_person_baseline_threshold_is_validation_selected_and_scales_are_labeled()
    test_design_evidence_and_training_code_are_hashed()
    test_visible_claims_match_canonical_artifacts_and_avoid_overclaiming()
    print("validation-design-audit-ok")
