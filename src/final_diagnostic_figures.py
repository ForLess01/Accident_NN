"""Classification diagnostics built from the frozen reference predictions.

These figures add no new model and no new fitting: they read the predictions the
canonical bundle already produced for 2024--2025 and answer questions that the
aggregate metrics leave open --- how separable the two classes actually are, why
the operating threshold sits where it sits, what the model is worth when used to
rank cases, and where it works better or worse.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
BASE_PATH = ROOT / "data" / "processed" / "base_limpia.parquet"

BASE_COLOR, ACCENT, GREY = "#496B7C", "#A63F12", "#7A8B94"
NEGATIVE, POSITIVE = "#8FA9B8", "#C9622A"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22,
})


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(TABLES_DIR / "final_reference_probabilities_2024_2025.csv")
    base = pd.read_parquet(BASE_PATH).reset_index(drop=True)
    segments = base.iloc[predictions["row_index"].values].reset_index(drop=True)
    observed = np.asarray(segments["target_multifatal"]).astype(int)
    if not (observed == predictions["actual_multifatal"].to_numpy(int)).all():
        raise RuntimeError("The frozen predictions no longer align with the base rows.")
    return predictions, segments


def separability_and_threshold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Panel 1: class overlap. Panel 2: the decision curve behind the threshold."""
    y = predictions["actual_multifatal"].to_numpy(int)
    probability = predictions["calibrated_probability"].to_numpy(float)
    threshold = float(predictions["calibrated_threshold"].iloc[0])

    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    sweep = pd.DataFrame([
        {
            "umbral": t,
            "precision": precision_score(y, probability >= t, zero_division=0),
            "recall": recall_score(y, probability >= t, zero_division=0),
            "f1": f1_score(y, probability >= t, zero_division=0),
            "alertas": int((probability >= t).sum()),
        }
        for t in grid
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    bins = np.linspace(0, 1, 41)
    axes[0].hist(probability[y == 0], bins=bins, color=NEGATIVE, alpha=0.85,
                 label=f"Un fallecido (n = {int((y == 0).sum()):,})".replace(",", " "))
    axes[0].hist(probability[y == 1], bins=bins, color=POSITIVE, alpha=0.85,
                 label=f"Dos o más (n = {int((y == 1).sum()):,})".replace(",", " "))
    axes[0].axvline(threshold, color="#202522", linestyle="--", linewidth=1.3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Probabilidad multifatal calibrada")
    axes[0].set_ylabel("Siniestros (escala log)")
    axes[0].set_title("Separación entre clases")
    axes[0].annotate(f"umbral {threshold:g}", xy=(threshold, axes[0].get_ylim()[1] * 0.35),
                     xytext=(threshold + 0.13, axes[0].get_ylim()[1] * 0.45), fontsize=8.5,
                     color="#334155", arrowprops=dict(arrowstyle="->", color=GREY))
    axes[0].legend(fontsize=8.5, loc="upper right")

    axes[1].plot(sweep["umbral"], sweep["precision"], color=BASE_COLOR, label="Precisión")
    axes[1].plot(sweep["umbral"], sweep["recall"], color=ACCENT, label="Recall")
    axes[1].plot(sweep["umbral"], sweep["f1"], color="#3F6B4E", linewidth=2, label="F1")
    axes[1].axvline(threshold, color="#202522", linestyle="--", linewidth=1.3)
    best = sweep.loc[sweep["f1"].idxmax()]
    axes[1].scatter([threshold], [float(sweep.loc[sweep["umbral"].eq(threshold), "f1"].iloc[0])],
                    s=70, color="#3F6B4E", zorder=5)
    axes[1].set_xlabel("Umbral sobre la probabilidad calibrada")
    axes[1].set_ylabel("Valor de la métrica")
    axes[1].set_title(f"Curva de decisión (F1 máximo en {best['umbral']:g})")
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Separabilidad de las clases y elección del umbral · referencia 2024–2025",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES_DIR / "fig18_separability_threshold.png", bbox_inches="tight")
    plt.close(fig)
    return sweep


def gain_and_segments(predictions: pd.DataFrame, segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel 1: value when the score is used to rank. Panel 2: where it holds."""
    y = predictions["actual_multifatal"].to_numpy(int)
    probability = predictions["calibrated_probability"].to_numpy(float)

    order = np.argsort(-probability)
    captured = np.cumsum(y[order]) / y.sum()
    reviewed = np.arange(1, len(y) + 1) / len(y)
    gain = pd.DataFrame({"revisado": reviewed, "capturado": captured})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(reviewed, captured, color=ACCENT, linewidth=2, label="Modelo")
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2, label="Orden aleatorio")
    for mark in (0.10, 0.20, 0.30):
        value = float(np.interp(mark, reviewed, captured))
        axes[0].vlines(mark, 0, value, color=BASE_COLOR, linewidth=1, alpha=0.6)
        axes[0].scatter([mark], [value], s=45, color=BASE_COLOR, zorder=5)
        axes[0].annotate(f"{value * 100:.0f}\\%".replace("\\", ""), (mark, value),
                         textcoords="offset points", xytext=(6, -12), fontsize=8.5, color="#334155")
    axes[0].set_xlabel("Fracción de siniestros revisada, ordenados por score")
    axes[0].set_ylabel("Fracción de multifatales capturada")
    axes[0].set_title("Ganancia acumulada")
    axes[0].legend(fontsize=8.5, loc="lower right")

    frame = pd.DataFrame({
        "y": y, "p": probability,
        "ZONA": segments["ZONA"].astype("string").fillna("SIN DATO").to_numpy(),
        "RED_VIAL": segments["RED_VIAL"].astype("string").fillna("SIN DATO").to_numpy(),
        "anio": pd.to_datetime(segments["FECHA"]).dt.year.to_numpy(),
    })
    rows: list[dict[str, Any]] = []
    for label, column in (("Zona", "ZONA"), ("Red vial", "RED_VIAL"), ("Año", "anio")):
        for value, group in frame.groupby(column):
            if len(group) < 100 or group["y"].nunique() < 2:
                continue
            rows.append({
                "dimension": label, "segmento": str(value), "n": len(group),
                "prevalencia": float(group["y"].mean()),
                "pr_auc": float(average_precision_score(group["y"], group["p"])),
            })
    by_segment = pd.DataFrame(rows)
    by_segment["mejora_sobre_prevalencia"] = by_segment["pr_auc"] / by_segment["prevalencia"]
    by_segment = by_segment.sort_values(["dimension", "mejora_sobre_prevalencia"], ascending=[True, True])

    labels = [f"{row.dimension}: {row.segmento}\n(n = {row.n})" for row in by_segment.itertuples()]
    positions = np.arange(len(by_segment))
    axes[1].barh(positions, by_segment["mejora_sobre_prevalencia"], color=BASE_COLOR, height=0.62)
    axes[1].axvline(1, color="#202522", linestyle="--", linewidth=1.2)
    for position, row in zip(positions, by_segment.itertuples()):
        axes[1].text(row.mejora_sobre_prevalencia + 0.05, position,
                     f"{row.mejora_sobre_prevalencia:.1f}×", va="center", fontsize=8.5, color="#334155")
    axes[1].set_yticks(positions, labels, fontsize=8)
    axes[1].set_xlabel("PR-AUC dividida por la prevalencia del segmento")
    axes[1].set_title("Mejora sobre la línea base, por segmento")
    axes[1].set_xlim(0, float(by_segment["mejora_sobre_prevalencia"].max()) * 1.22)

    fig.suptitle("Valor del ordenamiento y estabilidad por segmento · referencia 2024–2025",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES_DIR / "fig19_gain_segments.png", bbox_inches="tight")
    plt.close(fig)
    return gain, by_segment


def run() -> dict[str, Any]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    predictions, segments = load()
    sweep = separability_and_threshold(predictions)
    gain, by_segment = gain_and_segments(predictions, segments)

    sweep.to_csv(TABLES_DIR / "final_threshold_sweep_2024_2025.csv", index=False)
    by_segment.to_csv(TABLES_DIR / "final_segment_performance_2024_2025.csv", index=False)

    reviewed, captured = gain["revisado"].to_numpy(), gain["capturado"].to_numpy()
    summary = {
        "schema_version": 1,
        "generated_by": "src/final_diagnostic_figures.py",
        "source": "frozen canonical predictions; no model is fitted here",
        "captured_at_10_pct": float(np.interp(0.10, reviewed, captured)),
        "captured_at_20_pct": float(np.interp(0.20, reviewed, captured)),
        "captured_at_30_pct": float(np.interp(0.30, reviewed, captured)),
        "best_f1_threshold_on_reference": float(sweep.loc[sweep["f1"].idxmax(), "umbral"]),
        "operating_threshold": float(predictions["calibrated_threshold"].iloc[0]),
        "segment_lift_min": float(by_segment["mejora_sobre_prevalencia"].min()),
        "segment_lift_max": float(by_segment["mejora_sobre_prevalencia"].max()),
    }
    (TABLES_DIR / "final_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
