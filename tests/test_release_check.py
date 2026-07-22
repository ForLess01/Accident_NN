from __future__ import annotations
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.check_release import (
    PYTHON,
    RELEASE_PYTEST_ACTIVE_ENV,
    complete_pytest_command,
    find_changed_canonical_paths,
    find_forbidden_timeline_labels,
    find_missing_required_files,
    load_release_inventory,
    release_pytest_environment,
    verify_pdf_freshness,
)
import scripts.execute_notebooks as execute_notebooks

def test_inventory_is_explicit_and_complete() -> None:
    paths=load_release_inventory()
    assert paths==sorted(set(paths))
    assert not find_missing_required_files(ROOT,paths)
    required={'data/raw/source_manifest.json','docs/data_provenance.md','data/processed/demo_cases_manifest.json','report/build_manifest.json','scripts/execute_notebooks.py','scripts/build_report.py','src/source_provenance.py','src/temporal_diagnostics.py'}
    assert required<=set(paths)

def test_report_freshness_ignores_mtimes() -> None:
    verify_pdf_freshness()
    manifest=json.loads((ROOT/'report/build_manifest.json').read_text())
    source=ROOT/'report/main.tex'
    original=source.stat()
    before=(ROOT/'report/main.pdf').stat().st_mtime
    os.utime(source,(before+10000,before+10000))
    try: verify_pdf_freshness()
    finally: os.utime(source,ns=(original.st_atime_ns,original.st_mtime_ns))

def test_binary_modification_is_detected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        subprocess.run(['git','init','-q'],cwd=root,check=True)
        artifact=root/'artifact.bin'
        artifact.write_bytes(b'canonical\x00artifact')
        subprocess.run(['git','add','artifact.bin'],cwd=root,check=True)
        artifact.write_bytes(b'mutated\x00artifact')
        changed=find_changed_canonical_paths(root,['artifact.bin'])
        assert changed and any('artifact.bin' in entry for entry in changed)


def test_external_absolute_notebook_status_path_is_safe(tmp_path: Path,monkeypatch,capsys) -> None:
    external=tmp_path/'outside.ipynb'; external.write_text('{}',encoding='utf-8')
    monkeypatch.setattr(execute_notebooks,'execute',lambda path: None)
    monkeypatch.setattr(sys,'argv',['execute_notebooks.py',str(external)])
    assert execute_notebooks.main()==0
    payload=json.loads(capsys.readouterr().out)
    assert payload=={'notebook':str(external.resolve()),'status':'executed'}


def test_release_runs_complete_pytest_suite_with_nonrecursive_contract() -> None:
    assert complete_pytest_command()==[str(PYTHON),'-m','pytest','-q']
    environment=release_pytest_environment()
    assert environment[RELEASE_PYTEST_ACTIVE_ENV]=='1'


def test_forbidden_2023_selection_labels_are_detected_without_false_positive(tmp_path: Path) -> None:
    tables=tmp_path/'report/tables'; tables.mkdir(parents=True)
    protocol=tables/'protocol.md'
    protocol.write_text(
        'Selección de arquitectura: 2022. Validación de calibración y umbrales: 2023.',
        encoding='utf-8',
    )
    assert find_forbidden_timeline_labels(tmp_path)==[]
    protocol.write_text('Umbral: 2023 selection-period policy.',encoding='utf-8')
    hits=find_forbidden_timeline_labels(tmp_path)
    assert hits and any('selection-period' in hit for hit in hits)
    protocol.write_text('Los umbrales se seleccionan en 2023.',encoding='utf-8')
    assert find_forbidden_timeline_labels(tmp_path)
    protocol.write_text('Umbral seleccionado exclusivamente en validación 2023.',encoding='utf-8')
    assert find_forbidden_timeline_labels(tmp_path)
    protocol.write_text('La calibración se selecciona y ajusta exclusivamente en 2023.',encoding='utf-8')
    assert find_forbidden_timeline_labels(tmp_path)

if __name__=='__main__':
    test_inventory_is_explicit_and_complete(); test_report_freshness_ignores_mtimes(); test_binary_modification_is_detected(); print('release-check-ok')
