from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.model_protocol import PERSONAS_DERIVED_COLUMNS, fit_preprocessor, split_chronological, transform_features
from src.block_e_modeling import write_protocol_document
from src.source_provenance import verify_raw_sources

def test_feature_contract_excludes_personas() -> None:
    base=pd.read_parquet(ROOT/'data/processed/base_limpia.parquet')
    splits=split_chronological(base)
    scaler,encoders=fit_preprocessor(splits['X_train_raw'])
    transformed=transform_features(splits['X_validation_raw'].head(10),scaler,encoders)
    assert transformed.shape[1]==169
    assert PERSONAS_DERIVED_COLUMNS.isdisjoint(transformed.columns)
    assert pd.to_datetime(splits['X_train_raw']['FECHA']).dt.year.max()==2022
    assert pd.to_datetime(splits['X_validation_raw']['FECHA']).dt.year.eq(2023).all()
    assert pd.to_datetime(splits['X_test_raw']['FECHA']).dt.year.min()==2024

def test_raw_source_manifest_fails_closed_on_mutation() -> None:
    verified=verify_raw_sources()
    assert verified['source_count']==4
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); raw=root/'data/raw'; raw.mkdir(parents=True)
        source_manifest=json.loads((ROOT/'data/raw/source_manifest.json').read_text())
        for item in source_manifest['sources']:
            (raw/item['path']).write_bytes((ROOT/'data/raw'/item['path']).read_bytes())
        manifest=raw/'source_manifest.json'; manifest.write_text(json.dumps(source_manifest))
        (raw/source_manifest['sources'][0]['path']).write_bytes(b'tampered')
        try: verify_raw_sources(root=root,manifest_path=manifest)
        except RuntimeError as exc: assert 'integrity mismatch' in str(exc)
        else: raise AssertionError('mutated source was accepted')

def test_nested_temporal_roles_are_disjoint() -> None:
    roles=pd.read_csv(ROOT/'report/tables/temporal_nested_fold_roles.csv')
    assert set(roles.role)=={'fit','selection','calibration','outer'}
    for _, group in roles.groupby('fold'):
        ordered=group.set_index('role').loc[['fit','selection','calibration','outer']]
        assert all(
            pd.Timestamp(a) < pd.Timestamp(b)
            for a, b in zip(ordered.date_max.iloc[:-1], ordered.date_min.iloc[1:])
        )


def test_intermediate_protocol_writer_uses_canonical_chronology(tmp_path: Path) -> None:
    destination=tmp_path/'protocol.md'
    write_protocol_document(169,{'train':4872,'validation':2000},destination)
    text=destination.read_text(encoding='utf-8')
    assert 'Selección de arquitectura, configuración y semilla: ajuste en 2021 y comparación en 2022.' in text
    assert 'Validación de calibración y umbrales: exclusivamente 2023' in text
    assert 'se definen y ajustan únicamente en la partición de calibración y validación de umbrales de 2023' in text
    assert 'la arquitectura, la configuración y la semilla se seleccionan en 2022' in text
    assert 'Selección: 2023' not in text
    assert 'selection-period' not in text.lower()


def test_canonical_eda_ui_and_report_do_not_reintroduce_personas_predictors() -> None:
    descriptive=pd.read_csv(ROOT/'report/tables/tab01_descriptive_statistics.csv',index_col=0)
    assert PERSONAS_DERIVED_COLUMNS.isdisjoint(descriptive.index)
    summary=json.loads((ROOT/'report/tables/tab01_eda_summary.json').read_text())
    assert summary['missing_top_column'] not in PERSONAS_DERIVED_COLUMNS
    visible_sources=[ROOT/'app/streamlit_app.py',*sorted((ROOT/'report/sections').glob('*.tex'))]
    for path in visible_sources:
        content=path.read_text(encoding='utf-8')
        assert not any(label in content for label in PERSONAS_DERIVED_COLUMNS), path
    app=(ROOT/'app/streamlit_app.py').read_text(encoding='utf-8')
    assert 'vehículos y personas' not in app
    assert 'PERSONAS fue excluida' in app

if __name__=='__main__':
    test_feature_contract_excludes_personas(); test_raw_source_manifest_fails_closed_on_mutation(); test_nested_temporal_roles_are_disjoint(); test_canonical_eda_ui_and_report_do_not_reintroduce_personas_predictors(); print('model-protocol-ok')
