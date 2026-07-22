from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.final_explainability import EXPLANATION_RAW_COLUMNS,load_pre_endpoint_source,sha256_file

def test_explainability_boundary_and_stability() -> None:
    frame=load_pre_endpoint_source(ROOT/'data/processed/base_limpia.parquet')
    assert pd.to_datetime(frame.FECHA).max()<pd.Timestamp('2024-01-01')
    assert not any('persona' in value.lower() or 'edad_media' in value.lower() for value in EXPLANATION_RAW_COLUMNS)
    provenance=json.loads((ROOT/'report/tables/final_explainability_provenance.json').read_text())
    stability=pd.read_csv(ROOT/'report/tables/final_explainability_stability.csv')
    assert len(provenance['stability_seeds'])==3
    assert provenance['minimum_pairwise_spearman_rank_correlation']>0.9
    assert set(stability.stability_tier)<={'stable_top','stable_band','variable'}
    assert provenance['endpoint_2024_2025_data_loaded'] is False
    assert provenance['explanation_partition']=='explanation_validation_2023_only'

def test_explainability_hashes() -> None:
    manifest=json.loads((ROOT/'models/final/manifest.json').read_text())
    for relative,expected in manifest['explainability_artifact_hashes'].items(): assert sha256_file(ROOT/relative)==expected

if __name__=='__main__':
    test_explainability_boundary_and_stability(); test_explainability_hashes(); print('final-explainability-ok')
