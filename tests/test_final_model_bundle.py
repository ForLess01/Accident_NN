from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.final_model_bundle import (
    CanonicalModelBundle,
    _read_design_before_endpoint,
    _read_endpoint_after_freeze,
    sha256_file,
)
from src.model_protocol import PERSONAS_DERIVED_COLUMNS,split_chronological

def test_manifest_and_bundle_are_consistent() -> None:
    final=ROOT/'models/final'; manifest=json.loads((final/'manifest.json').read_text())
    schema=json.loads((final/'feature_schema.json').read_text()); features=json.loads((final/'feature_list.json').read_text())
    assert manifest['model_version']=='canonical-3.0.0'
    assert manifest['feature_count']==schema['processed_feature_count']==len(features)==169
    assert manifest['architecture']['hidden_units']==[64,32]
    assert manifest['architecture']['seed']==42
    assert manifest['calibration']['validation_partition']=='calibration_threshold_validation_2023_only'
    assert manifest['thresholds']['raw']['source_partition']=='threshold_validation_2023'
    assert manifest['thresholds']['calibrated']['source_partition']=='threshold_validation_2023_oof'
    assert PERSONAS_DERIVED_COLUMNS.isdisjoint(features)
    assert all(field['name'] not in PERSONAS_DERIVED_COLUMNS for field in schema['required_raw_fields'])
    for name,expected in manifest['artifact_hashes'].items(): assert sha256_file(final/name)==expected
    assert manifest['dataset']['raw_source_manifest_sha256']==sha256_file(ROOT/'data/raw/source_manifest.json')
    assert manifest['reference_evaluation']['confirmatory_independence'] is False
    assert manifest['reference_evaluation']['external_or_prospective_untouched_cohort'] is False
    gate=manifest['endpoint_open_gate']
    assert gate['design_rows_materialized_before_open']==manifest['splits']['train']['count']+manifest['splits']['validation']['count']
    assert gate['reference_rows_materialized_after_open']==manifest['splits']['reference']['count']
    claim=manifest['claims']['not_supported']
    assert '0.0252' in claim and '[0.00166, 0.04790]' in claim
    assert 'six paired intervals are nominal and unadjusted for multiplicity' in claim
    assert 'paired differences against Random Forest are not statistically significant' not in claim

def test_frozen_inference_reproduces_reference() -> None:
    base=pd.read_parquet(ROOT/'data/processed/base_limpia.parquet'); splits=split_chronological(base)
    runtime=CanonicalModelBundle(ROOT/'models/final',verify_hashes=True)
    predicted=runtime.predict_dataframe(splits['X_test_raw'])
    frozen=pd.read_csv(ROOT/'report/tables/final_reference_probabilities_2024_2025.csv')
    np.testing.assert_allclose(predicted.raw_probability,frozen.raw_probability,atol=1e-7,rtol=0)
    np.testing.assert_allclose(predicted.calibrated_probability,frozen.calibrated_probability,atol=1e-7,rtol=0)

def test_bootstrap_scope_is_explicit() -> None:
    meta=json.loads((ROOT/'report/tables/final_reference_bootstrap_metadata.json').read_text())
    assert meta['pipeline_refit_per_resample'] is False
    joined=' '.join(meta['excluded_uncertainty']).lower()
    for word in ['training','temporal','spatial','future','individual']: assert word in joined
    paired=json.loads((ROOT/'report/tables/final_paired_bootstrap_2024_2025.json').read_text())
    assert paired['comparison_family_size']==6
    assert paired['interval_coverage']=='nominal 95% per comparison'
    assert paired['multiplicity_adjustment']=='none'
    assert paired['simultaneous_familywise_coverage'] is False
    flags=[
        metric['nominal_significant_at_5pct']
        for baseline in paired['results'].values()
        for metric in baseline.values()
    ]
    assert len(flags)==6


def _minimal_partition(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'FECHA': pd.to_datetime(dates),
            'target_multifatal': [0, 1, 0][:len(dates)],
            'DEPARTAMENTO': ['LIMA'] * len(dates),
        },
        index=np.arange(100,100+len(dates)),
    )


def test_endpoint_reader_is_impossible_before_frozen_decisions(tmp_path: Path) -> None:
    calls: list[list[tuple[str,str,pd.Timestamp]]] = []
    def reader(_path: Path, *, filters: list[tuple[str,str,pd.Timestamp]]) -> pd.DataFrame:
        calls.append(filters)
        return _minimal_partition(['2021-02-01','2022-02-01','2023-02-01'])

    design,splits=_read_design_before_endpoint(tmp_path/'base.parquet',reader)
    assert pd.to_datetime(design.FECHA).dt.year.max()==2023
    assert len(splits['y_train'])==2 and len(splits['y_validation'])==1
    assert calls==[[('FECHA','<',pd.Timestamp('2024-01-01'))]]

    endpoint_calls=0
    def forbidden_reader(*_args: object,**_kwargs: object) -> pd.DataFrame:
        nonlocal endpoint_calls
        endpoint_calls+=1
        raise AssertionError('endpoint reader must not run')
    with pytest.raises(RuntimeError,match='complete frozen decision'):
        _read_endpoint_after_freeze(
            tmp_path/'base.parquet',tmp_path,{'thresholds.json':'x'},forbidden_reader
        )
    assert endpoint_calls==0


def test_endpoint_hook_precedes_materialization_and_filters_reference(tmp_path: Path) -> None:
    decision_names=['calibration_selection.json','calibrator.joblib','thresholds.json','feature_schema.json']
    hashes={}
    for name in decision_names:
        path=tmp_path/name; path.write_text(name,encoding='utf-8'); hashes[name]=sha256_file(path)
    opened=False
    filters_seen: list[tuple[str,str,pd.Timestamp]]=[]
    def hook() -> None:
        nonlocal opened
        opened=True
    def endpoint_reader(_path: Path, *, filters: list[tuple[str,str,pd.Timestamp]]) -> pd.DataFrame:
        assert opened, 'endpoint materialized before endpoint-open hook'
        filters_seen.extend(filters)
        return _minimal_partition(['2024-01-01','2025-12-31'])
    X,y=_read_endpoint_after_freeze(tmp_path/'base.parquet',tmp_path,hashes,endpoint_reader,hook)
    assert opened and len(X)==len(y)==2
    assert filters_seen==[
        ('FECHA','>=',pd.Timestamp('2024-01-01')),
        ('FECHA','<',pd.Timestamp('2026-01-01')),
    ]

if __name__=='__main__':
    test_manifest_and_bundle_are_consistent(); test_frozen_inference_reproduces_reference(); test_bootstrap_scope_is_explicit(); print('final-model-bundle-ok')
