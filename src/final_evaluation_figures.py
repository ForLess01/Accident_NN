"""Generate the definitive model-selection and reference-evaluation figures."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/accident_nn_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "report" / "tables"
FIGURES_DIR = ROOT / "report" / "figures"
COLORS = {"MLP_32_16": "#8CA3A3", "MLP_64_32": "#6F8090", "MLP_32": "#496B7C"}
SELECTED_COLOR = "#C94F16"


def selected_config_id() -> str:
    selection = json.loads((ROOT / "models" / "final" / "model_selection.json").read_text(encoding="utf-8"))
    return str(selection["selected_config"]["config_id"])


def leadership_summary(frame: pd.DataFrame) -> str:
    labels = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "f1_multifatal": "F1",
    }
    leaders = [f'{label}: {frame.loc[frame[metric].idxmax(), "label"]}' for metric, label in labels.items()]
    return "Liderazgo nominal - " + "; ".join(leaders) + ". No se afirma superioridad universal."


def generate_selection_robustness() -> Path:
    runs = pd.read_csv(TABLES_DIR / "model_selection_seed_grid_validation.csv")
    selected = selected_config_id()
    base_labels = {"MLP_32_16": "32-16", "MLP_64_32": "64-32", "MLP_32": "32"}
    labels = {
        config: f"{label} (seleccionada)" if config == selected else label
        for config, label in base_labels.items()
    }
    order = ["MLP_32_16", "MLP_64_32", "MLP_32"]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    offsets = np.linspace(-0.18, 0.18, len(runs["seed"].unique()))
    for x, config in enumerate(order):
        subset = runs[runs["config_id"] == config].sort_values("seed")
        ax.scatter(
            x + offsets,
            subset["pr_auc"],
            s=72,
            color=SELECTED_COLOR if config == selected else COLORS[config],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        median = float(subset["pr_auc"].median())
        ax.hlines(median, x - 0.31, x + 0.31, color="#202A27", linewidth=2.2)
        ax.text(x, median + 0.0022, f"mediana {median:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order)), [labels[value] for value in order])
    ax.set_ylabel("PR-AUC en selección 2023")
    ax.set_title("Robustez por configuración completa y semilla", loc="left", weight="bold")
    ax.grid(axis="y", color="#E1E5E2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / "final_seed_robustness.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def generate_model_evidence() -> Path:
    comparison = pd.read_csv(TABLES_DIR / "final_reference_baseline_comparison_2024_2025.csv")
    raw = comparison[comparison["probability_scale"] == "raw"].copy()
    raw["label"] = raw["model"].map(
        {
            "MLP_definitiva": "MLP definitiva",
            "LogisticRegression_balanced": "Regresión logística",
            "RandomForest_balanced": "Random Forest",
        }
    )
    raw = raw.dropna(subset=["label"])
    metrics = ["pr_auc", "roc_auc", "f1_multifatal"]
    titles = ["PR-AUC", "ROC-AUC", "F1 multifatal"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
    palette = ["#C94F16" if name == "MLP definitiva" else "#496B7C" for name in raw["label"]]
    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(raw["label"], raw[metric], color=palette, width=0.62)
        ax.set_title(title, weight="bold")
        ax.set_ylim(0, max(0.35, float(raw[metric].max()) * 1.18))
        ax.grid(axis="y", color="#E1E5E2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", rotation=24, labelsize=8.5)
        for bar, value in zip(bars, raw[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ax.get_ylim()[1] * 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Evidencia histórica 2024-2025: comparación en escala cruda", x=0.03, ha="left", weight="bold")
    fig.text(0.03, 0.012, leadership_summary(raw), fontsize=8.7, color="#5F6965")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    path = FIGURES_DIR / "final_model_evidence.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def generate_final_figures() -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [generate_selection_robustness(), generate_model_evidence()]
    outputs = {path.name: str(path.relative_to(ROOT)) for path in paths}
    manifest_path = ROOT / "models" / "final" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["evaluation_figure_hashes"] = {
        str(path.relative_to(ROOT)): digest(path) for path in paths
    }
    manifest["evaluation_figure_generator_sha256"] = digest(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    print(json.dumps(generate_final_figures(), ensure_ascii=False, indent=2))
