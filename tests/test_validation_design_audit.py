from __future__ import annotations
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.final_model_bundle import design_evidence_hashes,sha256_file

def test_design_evidence_and_one_network_conclusion() -> None:
    audit=json.loads((ROOT/'report/tables/design_validation_audit.json').read_text())
    strategies=pd.read_csv(ROOT/'report/tables/design_network_strategy_validation.csv')
    paired=pd.read_csv(ROOT/'report/tables/design_network_strategy_bootstrap.csv')
    assert audit['processed_features']==169 and audit['raw_input_fields']==21
    assert audit['personas_predictors_included'] is False
    assert set(strategies.strategy)=={'single_canonical_frozen','ensemble_mean_3_seeds','multibranch_context_vehicle_mean_3_seeds'}
    assert ((paired.ci_2_5<=0)&(paired.ci_97_5>=0)).all()

def test_design_hashes_match_manifest() -> None:
    manifest=json.loads((ROOT/'models/final/manifest.json').read_text())
    assert manifest['design_evidence_artifact_hashes']==design_evidence_hashes(ROOT)
    for name,expected in manifest['design_evidence_artifact_hashes'].items():
        path=ROOT/name if '/' in name else ROOT/'report/tables'/name
        assert sha256_file(path)==expected

if __name__=='__main__':
    test_design_evidence_and_one_network_conclusion(); test_design_hashes_match_manifest(); print('validation-design-audit-ok')
