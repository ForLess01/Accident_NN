from __future__ import annotations
import inspect,json,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import src.app_inference as inference
import app.streamlit_app as ui
import src.final_evaluation_figures as evaluation_figures

def test_schema_and_runtime_have_no_personas_inputs() -> None:
    schema=inference.load_feature_schema(); runtime=inference.load_prediction_stack()
    names={field['name'] for field in schema['required_raw_fields']}
    assert schema['processed_feature_count']==169==runtime.model.input_shape[-1]
    assert not {'n_personas','n_pasajeros','n_peatones','n_conductor_fugado','edad_media_involucrados'} & names
    summary=ui.canonical_design_summary(inference.load_manifest(),schema,ui.load_selection_runs())
    assert summary['trainable_parameters']==12993

def test_demo_roles_hash_and_prediction_parity() -> None:
    demos=inference.load_demo_cases()
    assert demos.role.tolist()==['TN','FP','boundary_FN','TP','synthetic_unseen_code']
    assert pd.to_datetime(demos.iloc[:4].FECHA).dt.year.eq(2023).all()
    assert pd.isna(demos.iloc[4].actual_multifatal)
    assert demos.iloc[4].CODIGO_VIA=='ZZ-UNSEEN'
    predicted=inference.predict_records(demos)
    np.testing.assert_allclose(predicted.calibrated_probability,demos.expected_calibrated_probability,atol=1e-7)


def test_selection_timeline_is_consistent_across_artifacts_and_visible_outputs() -> None:
    selection=inference.load_model_selection(); manifest=inference.load_manifest()
    timeline=inference.canonical_selection_timeline(selection,manifest)
    assert timeline=={
        'architecture_fit_period':'2021',
        'architecture_selection_period':'2022',
        'architecture_selection_label':'Fit 2021 / selección 2022',
        'final_refit_period':'2021–2022',
        'calibration_threshold_period':'2023',
        'reference_period':'2024–2025',
    }
    assert evaluation_figures.selection_metric_label()=='PR-AUC en selección 2022'
    source=inspect.getsource(ui.overview_page)
    assert 'Configuración completa, semilla, calibración y umbrales' not in source
    assert 'sin buscar arquitectura' in source

def test_coordinated_demo_csv_and_self_manifest_tampering_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); processed=root/'data/processed'; processed.mkdir(parents=True)
        csv_path=processed/'demo_cases.csv'; self_manifest_path=processed/'demo_cases_manifest.json'
        tampered=pd.read_csv(inference.DEMO_PATH)
        tampered.loc[0,'actual_multifatal']=1
        tampered.to_csv(csv_path,index=False)
        self_manifest=json.loads(inference.DEMO_MANIFEST_PATH.read_text())
        self_manifest['csv_sha256']=inference.sha256_file(csv_path)
        self_manifest_path.write_text(json.dumps(self_manifest),encoding='utf-8')
        monkeypatch.setattr(inference,'ROOT',root)
        monkeypatch.setattr(inference,'DEMO_PATH',csv_path)
        monkeypatch.setattr(inference,'DEMO_MANIFEST_PATH',self_manifest_path)
        inference.load_demo_cases.cache_clear()
        with pytest.raises(inference.RuntimeArtifactError,match='raíz de confianza'):
            inference.load_demo_cases()
    inference.load_demo_cases.cache_clear()


def test_demo_provenance_matches_only_untouched_payload() -> None:
    demo=inference.load_demo_cases().iloc[0]
    fields=[str(item['name']) for item in inference.load_feature_schema()['required_raw_fields']]
    untouched={field:demo[field] for field in fields}
    assert inference.demo_payload_matches_record(demo,untouched,fields)
    for field,value in {
        'FECHA':'2023-07-10',
        'HORA':'23:59',
        'CODIGO_VIA':'PE-1N',
        'n_vehiculos':int(demo['n_vehiculos'])+1,
    }.items():
        edited=untouched.copy(); edited[field]=value
        assert not inference.demo_payload_matches_record(demo,edited,fields)

def test_ui_uncertainty_and_proxy_copy_are_explicit() -> None:
    source=inspect.getsource(ui.evidence_page)+inspect.getsource(ui.estimate_page)
    assert 'Control de fuga por PERSONAS' in source
    assert 'verdad observada no disponible' in source
    assert 'Esto no implica causalidad' in source
    assert 'vehículos y personas' not in inspect.getsource(ui)

if __name__=='__main__':
    test_schema_and_runtime_have_no_personas_inputs(); test_demo_roles_hash_and_prediction_parity(); test_demo_provenance_matches_only_untouched_payload(); test_ui_uncertainty_and_proxy_copy_are_explicit(); print('app-inference-ok')
