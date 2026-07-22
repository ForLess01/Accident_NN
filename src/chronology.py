"""Single source of truth for the canonical temporal protocol.

The word *selection* is reserved for architecture, configuration, and seed
selection in 2022. The 2023 partition is used only to validate calibration
and decision thresholds; 2024--2025 is an already-observed historical
reference.
"""
from __future__ import annotations

from typing import Any, Mapping


ARCHITECTURE_SELECTION_PARTITION = "fit_2021_select_2022"
FINAL_REFIT_PARTITION = "2021-2022"
CALIBRATION_THRESHOLD_VALIDATION_PARTITION = "calibration_threshold_validation_2023_only"
RAW_THRESHOLD_VALIDATION_PARTITION = "threshold_validation_2023"
CALIBRATED_THRESHOLD_VALIDATION_PARTITION = "threshold_validation_2023_oof"
EXPLANATION_VALIDATION_PARTITION = "explanation_validation_2023_only"

CANONICAL_TIMELINE = {
    "architecture_fit_period": "2021",
    "architecture_selection_period": "2022",
    "architecture_selection_label": "Fit 2021 / selección 2022",
    "final_refit_period": "2021–2022",
    "calibration_threshold_period": "2023",
    "reference_period": "2024–2025",
}


def canonical_selection_timeline(
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Validate frozen temporal metadata and return its display chronology."""
    if selection.get("architecture_selection_partition") != ARCHITECTURE_SELECTION_PARTITION:
        raise ValueError("The architecture-selection partition is not canonical.")
    if selection.get("final_refit_partition") != FINAL_REFIT_PARTITION:
        raise ValueError("The final-refit partition is not canonical.")

    calibration = manifest.get("calibration", {})
    thresholds = manifest.get("thresholds", {})
    if not isinstance(calibration, Mapping) or not isinstance(thresholds, Mapping):
        raise ValueError("Calibration and threshold metadata must be mappings.")
    if calibration.get("validation_partition") != CALIBRATION_THRESHOLD_VALIDATION_PARTITION:
        raise ValueError("Calibration does not declare exclusive 2023 validation.")
    raw = thresholds.get("raw", {})
    calibrated = thresholds.get("calibrated", {})
    if not isinstance(raw, Mapping) or raw.get("source_partition") != RAW_THRESHOLD_VALIDATION_PARTITION:
        raise ValueError("The raw threshold does not declare 2023 threshold validation.")
    if (
        not isinstance(calibrated, Mapping)
        or calibrated.get("source_partition") != CALIBRATED_THRESHOLD_VALIDATION_PARTITION
    ):
        raise ValueError("The calibrated threshold does not declare OOF 2023 validation.")

    splits = manifest.get("splits", {})
    reference = splits.get("reference", {}) if isinstance(splits, Mapping) else {}
    if (
        not isinstance(reference, Mapping)
        or not str(reference.get("date_min", "")).startswith("2024-")
        or not str(reference.get("date_max", "")).startswith("2025-")
    ):
        raise ValueError("The historical reference is not 2024–2025.")
    return dict(CANONICAL_TIMELINE)


def protocol_chronology_markdown(
    *,
    train_count: int | None = None,
    validation_count: int | None = None,
) -> str:
    """Render the shared chronology used by every protocol document writer."""
    train_suffix = f" ({train_count} registros)" if train_count is not None else ""
    validation_suffix = f" ({validation_count} registros)" if validation_count is not None else ""
    return "\n".join(
        (
            "- Selección de arquitectura, configuración y semilla: ajuste en 2021 y comparación en 2022.",
            f"- Reajuste final de la MLP congelada: datos 2021--2022{train_suffix}.",
            (
                "- Validación de calibración y umbrales: exclusivamente 2023"
                f"{validation_suffix}; no se busca arquitectura en este periodo."
            ),
            (
                "- Referencia histórica 2024--2025: etiquetas ya observadas; "
                "no se usa para ajustar arquitectura, calibración ni umbrales."
            ),
        )
    )
