#!/usr/bin/env python3
"""Deterministic release gate for the canonical Accident_NN deliverable.

This command validates existing artifacts only. It never trains a model,
rebuilds evidence, or starts a persistent server.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DIRECT_CHECKS = [
    "tests/test_release_check.py",
    "tests/test_model_protocol.py",
    "tests/test_final_model_bundle.py",
    "tests/test_final_explainability.py",
    "tests/test_app_inference.py",
    "tests/test_validation_design_audit.py",
    "src/block_g_app_check.py",
]
INVENTORY_PATH = ROOT / "scripts" / "release_inventory.json"
FORBIDDEN_REFERENCES = (
    "models/v1",
    "models/v2",
    "model_v2_protocol",
    "block_e_modeling_v2",
    "test_v2_protocol",
    "letalidad_nn.keras",
    "calibration_posthoc",
)
TEXT_SUFFIXES = {".py", ".md", ".tex", ".json", ".toml", ".txt", ".csv", ".bib"}
STALE_NOTEBOOK_CLAIMS = ("tres arquitecturas", "no abre el periodo de referencia")


def announce(message: str) -> None:
    print(f"[release] {message}", flush=True)


def run(command: list[str]) -> None:
    announce("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_release_inventory(path: Path = INVENTORY_PATH) -> list[str]:
    """Load the explicit clean-clone inventory; globs are intentionally forbidden."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Falta el inventario explícito de entrega: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo leer el inventario de entrega: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("required_files"), list):
        raise RuntimeError("El inventario de entrega no cumple el esquema 1.")
    paths = [str(value) for value in payload["required_files"]]
    if not paths or len(paths) != len(set(paths)) or paths != sorted(paths):
        raise RuntimeError("El inventario debe ser no vacío, único y estar ordenado.")
    invalid = [value for value in paths if Path(value).is_absolute() or ".." in Path(value).parts]
    if invalid:
        raise RuntimeError("El inventario contiene rutas inseguras: " + ", ".join(invalid))
    return paths


def find_missing_required_files(root: Path, required_files: list[str]) -> list[str]:
    """Return every explicit inventory entry absent from a candidate clone root."""
    return [relative for relative in required_files if not (root / relative).is_file()]


def release_paths() -> list[str]:
    return load_release_inventory()


def require_paths() -> None:
    missing = find_missing_required_files(ROOT, release_paths())
    if missing:
        raise RuntimeError("Faltan artefactos canónicos: " + ", ".join(missing))


def verify_notebooks_and_python() -> None:
    python_files = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    python_files += [ROOT / "app" / "streamlit_app.py", ROOT / "scripts" / "check_release.py"]
    imported_modules: set[str] = set()
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if int(notebook.get("nbformat", 0)) != 4:
            raise RuntimeError(f"Notebook sin nbformat 4: {path.relative_to(ROOT)}")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source, filename=f"{path}:cell-{index}")
            except SyntaxError as exc:
                raise RuntimeError(f"Código inválido en {path.relative_to(ROOT)} celda {index}: {exc}") from exc
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module.split(".")[0])

    candidates = sorted(module for module in imported_modules if module != "src")
    probe = subprocess.run(
        [
            str(PYTHON),
            "-c",
            (
                "import importlib.util,json,sys; mods=json.loads(sys.argv[1]); "
                "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
            ),
            json.dumps(candidates),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    unresolved = json.loads(probe.stdout)
    if unresolved:
        raise RuntimeError("Imports no resolubles en el entorno: " + ", ".join(unresolved))


def find_stale_notebook_claims(root: Path = ROOT) -> list[str]:
    """Return obsolete methodological claims found in notebook markdown cells."""
    hits: list[str] = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", [])).lower()
            for claim in STALE_NOTEBOOK_CLAIMS:
                if claim in source:
                    hits.append(f"{path.relative_to(root)}:cell-{index} -> {claim}")
    return hits


def verify_notebook_claims() -> None:
    hits = find_stale_notebook_claims()
    if hits:
        raise RuntimeError("Afirmaciones metodológicas obsoletas en notebooks:\n" + "\n".join(hits))


def scan_forbidden_references() -> None:
    hits: list[str] = []
    roots = [ROOT / "app", ROOT / "src", ROOT / "tests", ROOT / "report", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "Plan.md"]
    for directory in roots:
        files.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES)
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in FORBIDDEN_REFERENCES:
            if token in text:
                hits.append(f"{path.relative_to(ROOT)} -> {token}")
    if hits:
        raise RuntimeError("Referencias obsoletas encontradas:\n" + "\n".join(hits))


def verify_pdf_freshness() -> None:
    pdf = ROOT / "report" / "main.pdf"
    sources = list((ROOT / "report").rglob("*.tex")) + list((ROOT / "report").rglob("*.bib"))
    if not pdf.exists() or pdf.stat().st_size < 10_000:
        raise RuntimeError("report/main.pdf falta o no parece un PDF final válido.")
    newer = [str(path.relative_to(ROOT)) for path in sources if path.stat().st_mtime > pdf.stat().st_mtime]
    if newer:
        raise RuntimeError("El PDF es anterior a sus fuentes: " + ", ".join(newer))


def verify_git_tracking(local_content: bool) -> None:
    untracked: list[str] = []
    for relative in release_paths():
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            untracked.append(relative)
    if not untracked:
        return
    message = "Archivos canónicos todavía no versionados: " + ", ".join(untracked)
    if local_content:
        announce("ADVERTENCIA: " + message)
    else:
        raise RuntimeError(message + ". Usá --local-content solo durante desarrollo.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-content",
        action="store_true",
        help="Valida el contenido local y muestra archivos no versionados como advertencia.",
    )
    args = parser.parse_args()
    if not PYTHON.exists():
        raise RuntimeError("No existe .venv/bin/python. Creá el entorno indicado en README.md.")

    require_paths()
    announce("artefactos requeridos presentes")
    for relative in DIRECT_CHECKS:
        run([str(PYTHON), relative])
    verify_notebooks_and_python()
    announce("notebooks, sintaxis e imports verificados")
    verify_notebook_claims()
    announce("afirmaciones metodológicas de notebooks verificadas")
    scan_forbidden_references()
    announce("sin referencias obsoletas")
    verify_pdf_freshness()
    announce("PDF final presente y actualizado")
    run(["git", "diff", "--check"])
    verify_git_tracking(args.local_content)
    announce("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[release] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
