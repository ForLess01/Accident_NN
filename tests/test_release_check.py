from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_release import find_missing_required_files, load_release_inventory


REPRESENTATIVE_REQUIRED_FILES = {
    "raw ONSV": "data/raw/BBDD_ONSV_SINIESTROS_FATALES_2021-2025.xlsx",
    "report source": "report/main.tex",
    "bibliography": "report/bib/referencias.bib",
    "model bundle": "models/final/model.keras",
    "processed runtime": "data/processed/base_limpia.parquet",
    "test": "tests/test_app_inference.py",
    "UI": "app/streamlit_app.py",
}


def test_explicit_inventory_is_complete_in_workspace() -> None:
    inventory = load_release_inventory()
    assert "scripts/release_inventory.json" in inventory
    assert "data/raw/Formato_2_Diccionario_de_datos.docx" in inventory
    assert "report/main.pdf" in inventory
    assert "docs/defensa_10min.md" in inventory
    assert "requirements-macos.txt" in inventory
    assert not find_missing_required_files(ROOT, inventory)


def test_representative_file_deletions_fail_completeness_check() -> None:
    inventory = load_release_inventory()
    for label, relative in REPRESENTATIVE_REQUIRED_FILES.items():
        assert relative in inventory, f"Representative {label} is absent from explicit inventory"

    with tempfile.TemporaryDirectory() as temporary_directory:
        clone = Path(temporary_directory)
        for relative in inventory:
            path = clone / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        assert not find_missing_required_files(clone, inventory)
        for label, relative in REPRESENTATIVE_REQUIRED_FILES.items():
            candidate = clone / relative
            candidate.unlink()
            missing = find_missing_required_files(clone, inventory)
            assert missing == [relative], f"Deleting representative {label} did not fail deterministically"
            candidate.touch()


if __name__ == "__main__":
    test_explicit_inventory_is_complete_in_workspace()
    test_representative_file_deletions_fail_completeness_check()
    print("release-check-ok")
