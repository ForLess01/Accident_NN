#!/usr/bin/env python3
"""Build the definitive report and seal content hashes (never mtimes)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MANIFEST = REPORT / "build_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_inputs() -> list[Path]:
    suffixes = {".tex", ".bib", ".png", ".csv", ".json", ".md"}
    paths = [
        path for path in REPORT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and path not in {MANIFEST}
    ]
    return sorted(paths, key=lambda value: str(value.relative_to(ROOT)))


def engine_metadata() -> dict[str, str]:
    version = subprocess.run(
        ["latexmk", "-v"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {"command": "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex", "version": version}


def build() -> dict[str, object]:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=REPORT,
        check=True,
    )
    pdf = REPORT / "main.pdf"
    if not pdf.is_file() or pdf.stat().st_size < 10_000:
        raise RuntimeError("report/main.pdf was not produced or is unexpectedly small.")
    inputs = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in canonical_inputs()
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "engine": engine_metadata(),
        "canonical_input_hashes": inputs,
        "pdf": {
            "path": "report/main.pdf",
            "bytes": pdf.stat().st_size,
            "sha256": sha256_file(pdf),
        },
        "freshness_rule": "content hashes only; checkout mtimes are irrelevant",
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
