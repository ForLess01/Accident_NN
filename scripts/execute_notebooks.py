#!/usr/bin/env python3
"""Execute narrative evidence notebooks without training the canonical model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path, root: Path = ROOT) -> str:
    """Use a repository-relative status path when possible, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def execute(path: Path, timeout: int = 600) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError(
            "Notebook execution requires nbformat and nbclient from requirements.txt."
        ) from exc
    notebook = nbformat.read(path, as_version=4)
    _, notebook = nbformat.validator.normalize(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    failures = [
        output
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]
    if failures:
        raise RuntimeError(f"Notebook contains failed cells after execution: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        candidate = path if path.is_absolute() else ROOT / path
        execute(candidate)
        print(json.dumps({"notebook": display_path(candidate), "status": "executed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
