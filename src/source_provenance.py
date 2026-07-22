"""Fail-closed provenance and target-proxy audits for official ONSV sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SOURCE_MANIFEST_PATH = RAW_DIR / "source_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_hash(path: Path = SOURCE_MANIFEST_PATH) -> str:
    return sha256_file(path)


def load_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), list):
        raise RuntimeError("data/raw/source_manifest.json does not satisfy schema version 1.")
    return payload


def verify_raw_sources(
    *, root: Path = ROOT, manifest_path: Path | None = None, reject_extra: bool = True
) -> dict[str, Any]:
    """Verify exact local source set, size and digest before any source is read."""
    path = manifest_path or root / "data" / "raw" / "source_manifest.json"
    payload = load_source_manifest(path)
    raw_dir = path.parent
    declared = {str(item["path"]): item for item in payload["sources"]}
    actual = {
        candidate.name
        for candidate in raw_dir.iterdir()
        if candidate.is_file() and candidate.name != path.name
    }
    missing = sorted(set(declared) - actual)
    extra = sorted(actual - set(declared))
    if missing or (reject_extra and extra):
        raise RuntimeError(f"Raw source set mismatch; missing={missing}, extra={extra}.")
    verified: list[dict[str, Any]] = []
    for name, metadata in sorted(declared.items()):
        source = raw_dir / name
        expected_size = int(metadata["bytes"])
        expected_hash = str(metadata["sha256"])
        actual_size = source.stat().st_size
        actual_hash = sha256_file(source)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise RuntimeError(
                f"Raw source integrity mismatch for {name}: "
                f"bytes {actual_size}!={expected_size} or sha256 {actual_hash}!={expected_hash}."
            )
        verified.append({"path": name, "bytes": actual_size, "sha256": actual_hash})
    return {
        "manifest_path": str(path.relative_to(root)),
        "manifest_sha256": sha256_file(path),
        "source_count": len(verified),
        "verified_sources": verified,
    }


def build_personas_proxy_audit(
    siniestros: pd.DataFrame,
    persons_path: Path,
    *, output_dir: Path | None = None,
    header_row: int = 4,
) -> dict[str, Any]:
    """Prove whether PERSONAS aggregates encode the multifatal target.

    PERSONAS is used only for lineage auditing. No PERSONAS-derived value is
    returned to the feature pipeline.
    """
    persons = pd.read_excel(persons_path, sheet_name=0, header=header_row, dtype=str)
    persons.columns = [str(column).strip() for column in persons.columns]
    persons["code"] = persons["CÓDIGO SINIESTRO"].astype("string").str.strip()
    severity = persons["GRAVEDAD"].astype("string").str.strip().str.upper()
    counts = persons.dropna(subset=["code"]).groupby("code").size().rename("person_row_count")
    deceased = (
        persons.assign(_deceased=severity.eq("FALLECIDO").astype("int8"))
        .dropna(subset=["code"])
        .groupby("code")["_deceased"]
        .sum()
        .rename("person_deceased_count")
    )
    base = siniestros.copy()
    base["code"] = base["CODIGO_SINIESTRO"].astype("string").str.strip()
    proof = base[["code", "FECHA", "FALLECIDOS", "target_multifatal"]].join(
        pd.concat([counts, deceased], axis=1), on="code"
    )
    proof["person_row_count"] = proof["person_row_count"].fillna(0).astype(int)
    proof["person_deceased_count"] = proof["person_deceased_count"].fillna(0).astype(int)
    proof["FALLECIDOS"] = pd.to_numeric(proof["FALLECIDOS"], errors="raise").astype(int)
    proof["target_rebuilt_from_personas"] = (proof["person_deceased_count"] >= 2).astype("int8")
    proof["deceased_equals_target_count"] = proof["person_deceased_count"].eq(proof["FALLECIDOS"])
    proof["rows_cover_deceased"] = proof["person_row_count"].ge(proof["FALLECIDOS"])
    proof["target_identity"] = proof["target_rebuilt_from_personas"].eq(proof["target_multifatal"])
    summary = {
        "schema_version": 1,
        "purpose": "executable target-proxy lineage proof; PERSONAS is forbidden as predictor source",
        "rows": int(len(proof)),
        "deceased_count_identity_rows": int(proof["deceased_equals_target_count"].sum()),
        "person_rows_cover_deceased_rows": int(proof["rows_cover_deceased"].sum()),
        "target_identity_rows": int(proof["target_identity"].sum()),
        "all_deceased_counts_equal_fallecidos": bool(proof["deceased_equals_target_count"].all()),
        "all_person_counts_cover_fallecidos": bool(proof["rows_cover_deceased"].all()),
        "all_targets_reconstructed": bool(proof["target_identity"].all()),
        "decision": "exclude every PERSONAS-derived aggregate from the canonical input contract",
        "personas_used_for_modeling": False,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        proof.to_csv(output_dir / "personas_target_proxy_identity.csv", index=False)
        (output_dir / "personas_target_proxy_identity.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary
