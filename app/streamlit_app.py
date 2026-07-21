"""Professional, evidence-first Streamlit interface for Accident_NN."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app_inference import (
    InputContractError,
    RuntimeArtifactError,
    departments_for_point,
    historical_comparison,
    load_clean_dataset,
    load_demo_cases,
    load_explainability_artifacts,
    load_feature_schema,
    load_input_options,
    load_known_road_codes,
    load_manifest,
    load_thresholds,
    mask_unsupported_regional_rates,
    predict_records,
    regional_summary,
    normalize_road_code,
    wilson_interval,
)
from src.final_model_bundle import sha256_file


st.set_page_config(
    page_title="Observatorio de multifatalidad vial — Perú",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ORANGE = "#A63F12"
INK = "#202522"
MUTED = "#5F6965"
BLUE = "#496B7C"
BLUE_LIGHT = "#B9CDD4"
SURFACE = "#F7F3EB"
GRID = "#E1E5E2"
UNSUPPORTED = "#C8CDCA"
MINIMUM_REGIONAL_SUPPORT = 30
INFERENCE_DATE_MIN = date(2021, 1, 1)
INFERENCE_DATE_MAX = date(2025, 12, 31)
SECTION_SLUGS = {
    "Panorama": "panorama",
    "Probar la red": "estimar",
    "Explorar datos": "explorar",
    "Patrones regionales": "regiones",
    "Evidencia del modelo": "evidencia",
}

APP_CSS = """
<style>
:root {
  --ink: #202522;
  --muted: #5F6965;
  --surface: #F7F3EB;
  --accent: #A63F12;
  --rule: #D9DED9;
}
html { scroll-behavior: smooth; }
body, .stApp { color: var(--ink); }
.stApp { background: #FCFBF7; }
.skip-link {
  position: fixed; left: 1rem; top: -5rem; z-index: 10000; background: var(--ink); color: #FFFFFF !important;
  padding: .7rem 1rem; text-decoration: none; border-radius: .2rem;
}
.skip-link:focus-visible { top: 1rem; }
.main .block-container { max-width: 1240px; padding-top: 1.25rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.018em; text-wrap: balance; scroll-margin-top: 5rem; }
h1 { max-width: 900px; }
p, li { text-wrap: pretty; }
code { color: #7D3213; background: #F4E9DE; border-radius: .25rem; }
.eyebrow {
  color: var(--accent); font-size: .76rem; font-weight: 800; letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: .5rem;
}
.scope-strip {
  border: 1px solid var(--rule); border-left: 5px solid var(--accent);
  background: var(--surface); padding: .9rem 1rem; margin: .6rem 0 1.25rem;
}
.scope-strip strong { color: var(--ink); }
.evidence-card {
  border-top: 3px solid var(--ink); background: #FFFFFF; padding: .85rem 1rem 1rem;
  min-height: 6.8rem; box-shadow: 0 1px 0 rgba(32,37,34,.08);
}
.evidence-label { color: var(--muted); font-size: .75rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.evidence-value { color: var(--ink); font-size: 1.45rem; font-weight: 750; line-height: 1.15; margin-top: .28rem; font-variant-numeric: tabular-nums; }
.evidence-detail { color: var(--muted); font-size: .8rem; margin-top: .32rem; }
.decision-panel {
  border: 1px solid var(--rule); border-top: 5px solid var(--accent); background: #FFFFFF;
  padding: 1rem 1.15rem; min-height: 9rem;
}
.decision-title { color: var(--ink); font-size: 1.35rem; font-weight: 800; margin: .3rem 0; }
.decision-copy { color: var(--muted); font-size: .92rem; }
.nn-pipeline {
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .55rem;
  margin: .8rem 0 1.4rem; align-items: stretch;
}
.nn-stage {
  position: relative; border: 1px solid var(--rule); border-top: 4px solid var(--ink);
  background: #FFFFFF; padding: .8rem; min-width: 0;
}
.nn-stage.accent { border-top-color: var(--accent); background: #FFF9F2; }
.nn-stage.external { border-top-color: #4F806A; background: #F4FAF6; }
.nn-stage:not(:last-child)::after {
  content: "→"; position: absolute; right: -.48rem; top: 42%; z-index: 2;
  color: var(--accent); font-weight: 900; background: #FCFBF7;
}
.nn-stage strong { display: block; font-size: .88rem; line-height: 1.2; }
.nn-stage span { display: block; color: var(--muted); font-size: .76rem; margin-top: .35rem; line-height: 1.35; }
.mono-value { font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stForm"] { border: 1px solid var(--rule); background: #FFFFFF; padding: 1rem; }
[data-testid="stRadio"] [role="radiogroup"] { gap: .35rem; border-bottom: 1px solid var(--rule); padding-bottom: .45rem; }
[data-testid="stRadio"] label { min-height: 44px; padding: .35rem .65rem; }
button:focus-visible, a:focus-visible, input:focus-visible, [role="radio"]:focus-visible {
  outline: 3px solid rgba(166,63,18,.52) !important; outline-offset: 2px;
}
.stButton button, [data-testid="stFormSubmitButton"] button { min-height: 44px; }
.stDownloadButton button { min-height: 44px; }
@media (max-width: 720px) {
  .main .block-container { padding-left: 1rem; padding-right: 1rem; }
  .evidence-card { min-height: auto; margin-bottom: .5rem; }
  .scope-strip { padding: .8rem; }
  .nn-pipeline { grid-template-columns: 1fr; }
  .nn-stage:not(:last-child)::after { content: "↓"; right: 50%; top: auto; bottom: -.72rem; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
</style>
"""


def _plot_layout(fig: go.Figure, *, height: int = 430, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=22, r=18, t=60, b=42),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Avenir Next, Source Sans 3, sans-serif", color=INK, size=13),
        showlegend=legend,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=INK),
        separators=",.",
    )
    # Styling the title of a figure that has none makes Plotly render the
    # literal string "undefined" as its gtitle.
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(size=18, color=INK))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def _percent(value: float, digits: int = 1) -> str:
    return format_number_es(value * 100, digits=digits, suffix=" %")


def format_number_es(value: float | int, *, digits: int = 0, suffix: str = "") -> str:
    """Format visible numbers consistently for Spanish (Peru)."""
    rendered = f"{float(value):,.{digits}f}"
    rendered = rendered.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")
    return f"{rendered}{suffix}"


def _compact_decimal_es(value: float, *, precision: int = 8) -> str:
    """Render an artifact decimal without inventing insignificant trailing zeros."""
    return np.format_float_positional(float(value), precision=precision, trim="-").replace(".", ",")


def overview_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    """Derive visible overview claims from the canonical manifest only."""
    metrics = manifest["reference_evaluation"]["metrics"]["calibrated"]
    class_rate = float(metrics["class_rate"])
    if not 0 < class_rate <= 1:
        raise RuntimeArtifactError("La prevalencia canónica debe estar en el intervalo (0, 1].")
    reference_count = int(metrics["n"])
    split_reference_count = int(manifest["splits"]["reference"]["count"])
    if reference_count != split_reference_count:
        raise RuntimeArtifactError("El tamaño de referencia no coincide entre las métricas y las particiones.")
    learning_rate = float(manifest["architecture"]["learning_rate"])
    if learning_rate <= 0:
        raise RuntimeArtifactError("La tasa de aprendizaje canónica debe ser positiva.")
    return {
        "class_rate_shorthand": f"1 de cada {max(1, round(1 / class_rate))}",
        "learning_rate": _compact_decimal_es(learning_rate),
        "reference_count": format_number_es(reference_count),
    }


STRATEGY_PRESENTATION = {
    "single_seed314_frozen": ("1 MLP congelada", ORANGE),
    "ensemble_mean_3_seeds": ("Ensemble 3 semillas", BLUE),
    "multibranch_162_context_13_companion_mean_3_seeds": ("Multirrama 162+13", "#7E9187"),
}


def strategy_presentation_table(strategies: pd.DataFrame) -> pd.DataFrame:
    """Join strategy labels/colors by stable artifact key, never by CSV row order."""
    if strategies["strategy"].duplicated().any() or set(strategies["strategy"]) != set(STRATEGY_PRESENTATION):
        raise RuntimeArtifactError("Las estrategias de diseño no coinciden con el contrato canónico.")
    ordered = strategies.set_index("strategy").loc[list(STRATEGY_PRESENTATION)].reset_index()
    ordered["label"] = ordered["strategy"].map(lambda key: STRATEGY_PRESENTATION[str(key)][0])
    ordered["color"] = ordered["strategy"].map(lambda key: STRATEGY_PRESENTATION[str(key)][1])
    return ordered


def strategy_ci_zero_summary(strategy_ci: pd.DataFrame) -> dict[str, int | str]:
    """Count intervals crossing zero directly from their persisted bounds."""
    required_metrics = ["pr_auc", "roc_auc", "f1_multifatal"]
    if strategy_ci["metric"].duplicated().any() or set(strategy_ci["metric"]) != set(required_metrics):
        raise RuntimeArtifactError("Los intervalos de estrategia no coinciden con el contrato canónico.")
    includes_zero = strategy_ci["ci_2_5"].le(0) & strategy_ci["ci_97_5"].ge(0)
    count = int(includes_zero.sum())
    total = int(len(strategy_ci))
    noun = "intervalo incluye" if total == 1 else "intervalos incluyen"
    return {"including_zero": count, "total": total, "copy": f"{count} de {total} {noun} 0"}


def person_strategy_labels(persons: pd.DataFrame, audit: dict[str, Any]) -> dict[str, str]:
    """Derive the count-rule label from the frozen validation-only threshold."""
    indexed = persons.set_index("model")
    required = {"regla_n_personas", "MLP_canónica_cruda"}
    if indexed.index.duplicated().any() or set(indexed.index) != required:
        raise RuntimeArtifactError("La comparación con n_personas no coincide con el contrato canónico.")
    threshold = float(audit["n_personas_rule_selected_on_validation"])
    if not np.isclose(float(indexed.loc["regla_n_personas", "threshold"]), threshold):
        raise RuntimeArtifactError("El umbral de n_personas no coincide entre las evidencias de diseño.")
    return {
        "regla_n_personas": f"Regla n_personas ≥ {format_number_es(threshold)}",
        "MLP_canónica_cruda": "MLP canónica cruda",
    }


def format_date_es(value: date | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.day:02d}/{timestamp.month:02d}/{timestamp.year:04d}"


def _card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f'<div class="evidence-card"><div class="evidence-label">{label}</div>'
        f'<div class="evidence-value">{value}</div><div class="evidence-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def _table_fallback(label: str, frame: pd.DataFrame, *, key: str | None = None) -> None:
    with st.expander(f"Ver tabla: {label}"):
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Descargar CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{(key or label).lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"download_{key or label}",
        )


@st.cache_data(show_spinner=False)
def cached_clean_dataset() -> pd.DataFrame:
    return load_clean_dataset().copy()


@st.cache_data(show_spinner=False)
def cached_regional_summary() -> pd.DataFrame:
    return regional_summary(MINIMUM_REGIONAL_SUPPORT)


TRACKING_DIMENSIONS = {
    "Clase de siniestro": "CLASE",
    "Zona": "ZONA",
    "Clima": "CLIMA",
    "Red vial": "RED_VIAL",
    "Tipo de vía": "TIPO_VIA",
    "Departamento": "DEPARTAMENTO",
}


@st.cache_data(show_spinner=False)
def cached_reference_with_features() -> pd.DataFrame:
    """Frozen 2024-2025 predictions joined with the registry context of each record."""
    probabilities = load_reference_artifacts()["probabilities"]
    base = cached_clean_dataset()
    context = base.loc[probabilities["row_index"].to_numpy(), list(TRACKING_DIMENSIONS.values())].reset_index(drop=True)
    joined = pd.concat([probabilities.reset_index(drop=True), context], axis=1)
    for column in TRACKING_DIMENSIONS.values():
        joined[column] = joined[column].fillna("SIN DATO").astype(str)
    return joined


@st.cache_data(show_spinner=False)
def load_reference_artifacts() -> dict[str, Any]:
    tables = ROOT / "report" / "tables"
    required = {
        "metrics": tables / "final_reference_metrics_2024_2025.json",
        "baseline": tables / "final_reference_baseline_comparison_2024_2025.csv",
        "ci": tables / "final_reference_bootstrap_ci_2024_2025.csv",
        "confusion": tables / "final_reference_confusion_matrix_2024_2025.csv",
        "probabilities": tables / "final_reference_probabilities_2024_2025.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeArtifactError(f"Faltan evidencias canónicas: {', '.join(missing)}")
    expected_hashes = load_manifest().get("reference_artifact_hashes", {})
    for path in required.values():
        expected = expected_hashes.get(path.name)
        if not expected or sha256_file(path) != expected:
            raise RuntimeArtifactError(
                f"La evidencia {path.name} no coincide con el manifiesto canónico. Restaurá el artefacto antes de abrirla."
            )
    try:
        return {
            "metrics": json.loads(required["metrics"].read_text(encoding="utf-8")),
            "baseline": pd.read_csv(required["baseline"]),
            "ci": pd.read_csv(required["ci"]),
            "confusion": pd.read_csv(required["confusion"]),
            "probabilities": pd.read_csv(required["probabilities"]),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(f"No se pudieron leer las evidencias canónicas: {exc}") from exc


@st.cache_data(show_spinner=False)
def load_design_artifacts() -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Load validation-only design evidence and verify every persisted hash."""
    tables = ROOT / "report" / "tables"
    required = {
        "audit": tables / "design_validation_audit.json",
        "regularization": tables / "design_regularization_summary.csv",
        "strategies": tables / "design_network_strategy_validation.csv",
        "strategy_bootstrap": tables / "design_network_strategy_bootstrap.csv",
        "persons": tables / "design_n_personas_reference_comparison.csv",
        "persons_bootstrap": tables / "design_n_personas_paired_bootstrap.csv",
        "annual": tables / "design_annual_stability_2024_2025.csv",
    }
    expected = load_manifest().get("design_evidence_artifact_hashes", {})
    for path in required.values():
        if not path.exists() or expected.get(path.name) != sha256_file(path):
            raise RuntimeArtifactError(f"La evidencia de diseño {path.name} falta o no coincide con el manifiesto.")
    return {
        "audit": json.loads(required["audit"].read_text(encoding="utf-8")),
        **{key: pd.read_csv(path) for key, path in required.items() if key != "audit"},
    }


@st.cache_data(show_spinner=False)
def load_selection_runs() -> pd.DataFrame:
    """Load the frozen configuration-by-seed grid with manifest verification."""
    path = ROOT / "report" / "tables" / "model_selection_seed_grid_validation.csv"
    expected = load_manifest().get("selection_artifact_hashes", {}).get(path.name)
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise RuntimeArtifactError("La grilla de selección no coincide con el manifiesto canónico.")
    return pd.read_csv(path)


def canonical_design_summary(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    selection_runs: pd.DataFrame,
) -> dict[str, Any]:
    """Derive every visible network count from frozen canonical artifacts."""
    raw_input_fields = len(schema["required_raw_fields"])
    processed_features = int(schema["processed_feature_count"])
    hidden_units = [int(value) for value in manifest["architecture"]["hidden_units"]]
    if processed_features != int(manifest["feature_count"]) or not hidden_units:
        raise RuntimeArtifactError("El esquema y la arquitectura canónica no coinciden.")
    widths = [processed_features, *hidden_units, 1]
    trainable_parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
    configuration_count = int(selection_runs["config_id"].nunique())
    seed_count = int(selection_runs["seed"].nunique())
    run_count = int(len(selection_runs[["config_id", "seed"]].drop_duplicates()))
    if run_count != configuration_count * seed_count:
        raise RuntimeArtifactError("La grilla configuración×semilla está incompleta.")
    return {
        "raw_input_fields": raw_input_fields,
        "processed_features": processed_features,
        "hidden_units": hidden_units,
        "hidden_layer_count": len(hidden_units),
        "dense_layer_count": len(hidden_units) + 1,
        "trainable_parameters": trainable_parameters,
        "configuration_count": configuration_count,
        "seed_count": seed_count,
        "run_count": run_count,
    }


def app_header(manifest: dict[str, Any]) -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown('<a class="skip-link" href="#main-content">Saltar al contenido principal</a><div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Observatorio académico · Seguridad vial · Perú</div>', unsafe_allow_html=True)
    st.title("Multifatalidad en siniestros viales ya fatales")
    st.markdown(
        '<div class="scope-strip"><strong>¿Qué hace este sistema?</strong> Clasifica retrospectivamente un siniestro vial fatal registrado en Perú '
        'como <strong>1 fallecido</strong> o <strong>2+ fallecidos</strong> y muestra la probabilidad calibrada, la evidencia y sus límites. Fuente: ONSV 2021–2025.</div>',
        unsafe_allow_html=True,
    )


def _monthly_tracking_chart(probabilities: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    frame = probabilities.copy()
    frame["mes"] = pd.to_datetime(frame["fecha"]).dt.to_period("M").dt.to_timestamp()
    monthly = (
        frame.groupby("mes")
        .agg(
            observada=("actual_multifatal", "mean"),
            predicha=("calibrated_probability", "mean"),
            n=("actual_multifatal", "size"),
            multifatales=("actual_multifatal", "sum"),
        )
        .reset_index()
    )
    # Same support policy as the rest of the app: months below the minimum
    # would display noise as if it were signal.
    monthly = monthly[monthly["n"] >= MINIMUM_REGIONAL_SUPPORT].copy()
    monthly["observada_pct"] = monthly["observada"] * 100
    monthly["predicha_pct"] = monthly["predicha"] * 100
    intervals = [wilson_interval(int(positives), int(n)) for positives, n in zip(monthly["multifatales"], monthly["n"])]
    monthly["ci_inf_pct"] = [low * 100 for low, _ in intervals]
    monthly["ci_sup_pct"] = [high * 100 for _, high in intervals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["mes"], y=monthly["observada_pct"], mode="lines+markers", name="Tasa multifatal observada",
        line=dict(color=INK, width=2.2), marker=dict(size=7, color=INK),
        error_y=dict(
            type="data",
            array=monthly["ci_sup_pct"] - monthly["observada_pct"],
            arrayminus=monthly["observada_pct"] - monthly["ci_inf_pct"],
            color="rgba(32,37,34,.35)",
            thickness=1.2,
        ),
        customdata=monthly[["n", "multifatales", "ci_inf_pct", "ci_sup_pct"]],
        hovertemplate="%{x|%b %Y}<br>Observada: %{y:.1f}%<br>Multifatales: %{customdata[1]:,} de %{customdata[0]:,}<br>IC 95%%: %{customdata[2]:.1f}–%{customdata[3]:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["mes"], y=monthly["predicha_pct"], mode="lines+markers", name="Probabilidad media del modelo",
        line=dict(color=ORANGE, width=2.2, dash="dot"), marker=dict(size=7, color=ORANGE, symbol="diamond"),
        hovertemplate="%{x|%b %Y}<br>Predicha media: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Mes a mes en 2024–2025: lo que el modelo estimó vs. lo que ocurrió")
    fig.update_xaxes(title="Mes")
    fig.update_yaxes(title="Multifatalidad (%)", rangemode="tozero")
    table = monthly.assign(mes=monthly["mes"].dt.strftime("%Y-%m"))[
        ["mes", "n", "multifatales", "observada_pct", "predicha_pct", "ci_inf_pct", "ci_sup_pct"]
    ].round(2)
    return _plot_layout(fig, height=430, legend=True), table


def _risk_ordering_chart(probabilities: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    frame = probabilities.copy()
    frame["quintil"] = pd.qcut(
        frame["calibrated_probability"].rank(method="first"), 5,
        labels=["Q1 · score más bajo", "Q2", "Q3", "Q4", "Q5 · score más alto"],
    )
    grouped = (
        frame.groupby("quintil", observed=True)
        .agg(
            observada=("actual_multifatal", "mean"),
            predicha=("calibrated_probability", "mean"),
            n=("actual_multifatal", "size"),
            multifatales=("actual_multifatal", "sum"),
        )
        .reset_index()
    )
    grouped["observada_pct"] = grouped["observada"] * 100
    grouped["predicha_pct"] = grouped["predicha"] * 100
    intervals = [wilson_interval(int(positives), int(n)) for positives, n in zip(grouped["multifatales"], grouped["n"])]
    grouped["ci_inf_pct"] = [low * 100 for low, _ in intervals]
    grouped["ci_sup_pct"] = [high * 100 for _, high in intervals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["quintil"].astype(str), y=grouped["observada_pct"], name="Tasa multifatal observada",
        marker_color=[BLUE_LIGHT, BLUE_LIGHT, BLUE, BLUE, ORANGE],
        error_y=dict(type="data", array=grouped["ci_sup_pct"] - grouped["observada_pct"], arrayminus=grouped["observada_pct"] - grouped["ci_inf_pct"], color=MUTED),
        customdata=grouped[["n", "multifatales", "predicha_pct"]],
        hovertemplate="<b>%{x}</b><br>Observada: %{y:.1f}%<br>Predicha media: %{customdata[2]:.1f}%<br>Multifatales: %{customdata[1]:,} de %{customdata[0]:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["quintil"].astype(str), y=grouped["predicha_pct"], mode="markers", name="Probabilidad media del modelo",
        marker=dict(color=INK, symbol="diamond", size=12, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Predicha media: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="Cinco grupos según el score del modelo: la multifatalidad real crece con el score")
    fig.update_xaxes(title="Registros de 2024–2025 ordenados por el modelo, en quintiles")
    fig.update_yaxes(title="Multifatalidad (%)", rangemode="tozero")
    table = grouped[["quintil", "n", "multifatales", "observada_pct", "predicha_pct", "ci_inf_pct", "ci_sup_pct"]].round(2)
    return _plot_layout(fig, height=430, legend=True), table


def _category_tracking_chart(joined: pd.DataFrame, column: str, label: str) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        joined.groupby(column)
        .agg(
            observada=("actual_multifatal", "mean"),
            predicha=("calibrated_probability", "mean"),
            n=("actual_multifatal", "size"),
            multifatales=("actual_multifatal", "sum"),
        )
        .reset_index()
        .rename(columns={column: "categoria"})
    )
    grouped = grouped[grouped["n"] >= MINIMUM_REGIONAL_SUPPORT].sort_values("observada")
    grouped["observada_pct"] = grouped["observada"] * 100
    grouped["predicha_pct"] = grouped["predicha"] * 100
    intervals = [wilson_interval(int(positives), int(n)) for positives, n in zip(grouped["multifatales"], grouped["n"])]
    grouped["ci_inf_pct"] = [low * 100 for low, _ in intervals]
    grouped["ci_sup_pct"] = [high * 100 for _, high in intervals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["observada_pct"], y=grouped["categoria"], mode="markers", name="Tasa multifatal observada",
        marker=dict(color=INK, size=11),
        error_x=dict(type="data", array=grouped["ci_sup_pct"] - grouped["observada_pct"], arrayminus=grouped["observada_pct"] - grouped["ci_inf_pct"], color=MUTED, thickness=1.3),
        customdata=grouped[["n", "multifatales"]],
        hovertemplate="<b>%{y}</b><br>Observada: %{x:.1f}%<br>Multifatales: %{customdata[1]:,} de %{customdata[0]:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["predicha_pct"], y=grouped["categoria"], mode="markers", name="Probabilidad media del modelo",
        marker=dict(color=ORANGE, symbol="diamond", size=12, line=dict(color="#FFFFFF", width=1.2)),
        hovertemplate="<b>%{y}</b><br>Predicha media: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(title=f"Por {label.lower()}: lo que el modelo estimó vs. lo que ocurrió en 2024–2025")
    fig.update_xaxes(title="Multifatalidad (%)", rangemode="tozero")
    fig.update_yaxes(title=None)
    height = max(360, 90 + 42 * len(grouped))
    table = grouped[["categoria", "n", "multifatales", "observada_pct", "predicha_pct", "ci_inf_pct", "ci_sup_pct"]].round(2)
    return _plot_layout(fig, height=height, legend=True), table


def _category_monthly_chart(joined: pd.DataFrame, column: str, category: str, label: str) -> tuple[go.Figure | None, pd.DataFrame]:
    subset = joined[joined[column] == category].copy()
    subset["mes"] = pd.to_datetime(subset["fecha"]).dt.to_period("M").dt.to_timestamp()
    monthly = (
        subset.groupby("mes")
        .agg(
            observada=("actual_multifatal", "mean"),
            predicha=("calibrated_probability", "mean"),
            n=("actual_multifatal", "size"),
            multifatales=("actual_multifatal", "sum"),
        )
        .reset_index()
    )
    monthly = monthly[monthly["n"] >= MINIMUM_REGIONAL_SUPPORT].copy()
    if len(monthly) < 4:
        return None, pd.DataFrame()
    monthly["observada_pct"] = monthly["observada"] * 100
    monthly["predicha_pct"] = monthly["predicha"] * 100
    intervals = [wilson_interval(int(positives), int(n)) for positives, n in zip(monthly["multifatales"], monthly["n"])]
    monthly["ci_inf_pct"] = [low * 100 for low, _ in intervals]
    monthly["ci_sup_pct"] = [high * 100 for _, high in intervals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["mes"], y=monthly["observada_pct"], mode="lines+markers", name="Tasa multifatal observada",
        line=dict(color=INK, width=2.2), marker=dict(size=7, color=INK),
        error_y=dict(type="data", array=monthly["ci_sup_pct"] - monthly["observada_pct"], arrayminus=monthly["observada_pct"] - monthly["ci_inf_pct"], color="rgba(32,37,34,.35)", thickness=1.2),
        customdata=monthly[["n", "multifatales"]],
        hovertemplate="%{x|%b %Y}<br>Observada: %{y:.1f}%<br>Multifatales: %{customdata[1]:,} de %{customdata[0]:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["mes"], y=monthly["predicha_pct"], mode="lines+markers", name="Probabilidad media del modelo",
        line=dict(color=ORANGE, width=2.2, dash="dot"), marker=dict(size=7, color=ORANGE, symbol="diamond"),
        hovertemplate="%{x|%b %Y}<br>Predicha media: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title=f"{label}: {category} · seguimiento mensual del modelo frente a la realidad")
    fig.update_xaxes(title="Mes")
    fig.update_yaxes(title="Multifatalidad (%)", rangemode="tozero")
    table = monthly.assign(mes=monthly["mes"].dt.strftime("%Y-%m"))[
        ["mes", "n", "multifatales", "observada_pct", "predicha_pct", "ci_inf_pct", "ci_sup_pct"]
    ].round(2)
    return _plot_layout(fig, height=420, legend=True), table


def overview_page(manifest: dict[str, Any]) -> None:
    st.header("Panorama")
    metrics = manifest["reference_evaluation"]["metrics"]["calibrated"]
    provenance = overview_provenance(manifest)
    columns = st.columns(3)
    overview_cards = [
        ("Siniestros fatales estudiados", format_number_es(manifest["dataset"]["row_count"]), "Registro oficial ONSV, 2021–2025"),
        ("¿Cuántos son multifatales?", provenance["class_rate_shorthand"], f'{_percent(metrics["class_rate"])} de los siniestros fatales deja 2+ fallecidos'),
        ("¿El modelo separa las clases?", f'ROC-AUC {format_number_es(metrics["roc_auc"], digits=3)}', "Referencia histórica 2024–2025; segunda consulta declarada"),
    ]
    for column, card in zip(columns, overview_cards):
        with column:
            _card(*card)

    st.subheader("Cómo funciona la red neuronal")
    architecture = manifest["architecture"]
    design = canonical_design_summary(manifest, load_feature_schema(), load_selection_runs())
    hidden_units = design["hidden_units"]
    threshold = manifest["thresholds"]["calibrated"]["value"]
    st.markdown(
        f"""
        <div class="nn-pipeline" role="img" aria-label="Flujo de {design['raw_input_fields']} campos crudos a una clasificación multifatal">
          <div class="nn-stage"><strong>{design['raw_input_fields']} campos</strong><span>Evento, ubicación, vía, entorno, vehículos y personas</span></div>
          <div class="nn-stage"><strong>{design['processed_features']} features</strong><span>Escalado, ciclos temporales, categorías e interacciones</span></div>
          <div class="nn-stage accent"><strong>Dense {hidden_units[0]} + ReLU</strong><span>L2 {architecture['l2']:.0e} · Dropout {architecture['dropout']:.2f} durante entrenamiento</span></div>
          <div class="nn-stage accent"><strong>Dense {hidden_units[1]} + ReLU</strong><span>L2 {architecture['l2']:.0e} · Dropout {architecture['dropout']:.2f} durante entrenamiento</span></div>
          <div class="nn-stage external"><strong>Platt externo</strong><span>σ(a·logit(s)+b); calibra el score congelado</span></div>
          <div class="nn-stage"><strong>Clase estimada</strong><span>1 fallecido / 2+ fallecidos · umbral {threshold:.2f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"La MLP tiene {design['hidden_layer_count']} capas ocultas, {design['dense_layer_count']} capas densas entrenables "
        f"y {format_number_es(design['trainable_parameters'])} parámetros. La búsqueda comparó "
        f"{design['configuration_count']} configuraciones completas × {design['seed_count']} semillas = "
        f"{design['run_count']} corridas y congeló {architecture['config_id']}, semilla {architecture['seed']}."
    )
    technique_table = pd.DataFrame([
        {"Fase": "Representación", "Técnica": "Escalado + one-hot + codificación cíclica", "Decisión": "Ajustada solo con 2021–2022"},
        {"Fase": "Red", "Técnica": "ReLU, L2 y dropout", "Decisión": "Capacidad compacta; evidencia de ablación en 2023"},
        {"Fase": "Optimización", "Técnica": "BCE ponderada + Adam", "Decisión": f"Pesos de clase; LR inicial {provenance['learning_rate']}"},
        {"Fase": "Control", "Técnica": "Early stopping + ReduceLROnPlateau", "Decisión": "Monitor PR-AUC de validación"},
        {"Fase": "Postproceso", "Técnica": "Platt OOF", "Decisión": "Externa a la NN; no es stacking"},
    ])
    _table_fallback("técnicas por fase", technique_table, key="tecnicas_por_fase")

    st.subheader("La demostración")
    st.write(
        f"Estas lecturas usan los {provenance['reference_count']} siniestros de 2024–2025 como referencia histórica ya consultada. "
        "Como expectativa promedio, una calibración útil acerca la probabilidad media a la frecuencia observada; "
        "cada mes puede apartarse por variabilidad muestral, especialmente cuando reúne pocos registros."
    )
    try:
        probabilities = load_reference_artifacts()["probabilities"]
        tracking_fig, tracking_table = _monthly_tracking_chart(probabilities)
        st.plotly_chart(tracking_fig, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Cada punto de la línea sólida es la tasa multifatal realmente observada ese mes; la línea punteada es la "
            "probabilidad media que el modelo asignó a esos mismos registros. Su cercanía agregada es compatible con buena calibración; no garantiza coincidencia punto a punto. "
            f"Las barras verticales son intervalos de Wilson al 95 %: los meses con pocos registros oscilan dentro de ese margen. "
            f"Se muestran los meses con al menos {MINIMUM_REGIONAL_SUPPORT} registros; el cierre de 2025 es preliminar."
        )
        _table_fallback("seguimiento mensual predicho vs. observado", tracking_table, key="seguimiento_mensual")

        ordering_fig, ordering_table = _risk_ordering_chart(probabilities)
        st.plotly_chart(ordering_fig, width="stretch", config={"displayModeBar": False})
        top = ordering_table.iloc[-1]
        bottom = ordering_table.iloc[0]
        st.caption(
            f"El quintil de mayor score concentra una tasa multifatal de {format_number_es(top['observada_pct'], digits=1, suffix=' %')} "
            f"frente a {format_number_es(bottom['observada_pct'], digits=1, suffix=' %')} en el de menor score: el modelo separa las clases en esta referencia histórica. "
            "Las barras incluyen intervalos de Wilson al 95 %."
        )
        _table_fallback("multifatalidad observada por quintil de score", ordering_table, key="quintiles_score")

        st.subheader("Elegí una categoría y mirá cómo predice el modelo")
        st.write(
            "La misma comparación, desagregada: para cada categoría del registro, la tasa multifatal que ocurrió en "
            "2024–2025 frente a la probabilidad media que el modelo asignó. Solo se muestran categorías con al menos "
            f"{MINIMUM_REGIONAL_SUPPORT} registros."
        )
        joined = cached_reference_with_features()
        dimension_label = st.selectbox("Dimensión del registro", list(TRACKING_DIMENSIONS), index=0, key="tracking_dimension")
        dimension_column = TRACKING_DIMENSIONS[dimension_label]
        category_fig, category_table = _category_tracking_chart(joined, dimension_column, dimension_label)
        st.plotly_chart(category_fig, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Cuando el rombo naranja cae dentro del intervalo del punto negro, existe compatibilidad visual entre promedio predicho y frecuencia observada. "
            "No es una prueba de calibración puntual. Intervalos de Wilson al 95 %."
        )
        _table_fallback("predicho vs. observado por categoría", category_table, key="categoria_tracking")

        if not category_table.empty:
            drill_options = category_table.sort_values("n", ascending=False)["categoria"].tolist()
            drill_category = st.selectbox(
                f"Seguimiento mensual para una categoría de {dimension_label.lower()}",
                drill_options,
                index=0,
                key="tracking_category",
            )
            drill_fig, drill_table = _category_monthly_chart(joined, dimension_column, drill_category, dimension_label)
            if drill_fig is None:
                st.info(
                    f"«{drill_category}» no reúne suficientes meses con n ≥ {MINIMUM_REGIONAL_SUPPORT} para una serie mensual "
                    "estable; la comparación agregada de arriba sigue siendo válida."
                )
            else:
                st.plotly_chart(drill_fig, width="stretch", config={"displayModeBar": False})
                _table_fallback("seguimiento mensual de la categoría", drill_table, key="categoria_mensual")
    except RuntimeArtifactError as exc:
        st.warning(f"No se pudo cargar la evidencia congelada para la demostración: {exc}")

    with st.expander("Diseño experimental y garantías metodológicas"):
        protocol = pd.DataFrame(
            [
                {"Etapa": "Entrenamiento", "Periodo": "2021–2022", "n": manifest["splits"]["train"]["count"], "Uso": "Ajuste de pesos y preprocesamiento"},
                {"Etapa": "Selección", "Periodo": "2023", "n": manifest["splits"]["validation"]["count"], "Uso": "Configuración completa, semilla, calibración y umbrales"},
                {"Etapa": "Referencia histórica", "Periodo": "2024–2025", "n": manifest["splits"]["reference"]["count"], "Uso": "Evaluación descriptiva; sin ajustes"},
            ]
        )
        st.dataframe(protocol, width="stretch", hide_index=True)
        st.markdown(
            f"""
            - **Sin fuga de resultado:** el modelo nunca ve fallecidos, lesionados, vehículos dañados ni causas de la investigación posterior.
            - **Orden cronológico real:** se entrena con el pasado y se describe el desempeño en periodos posteriores.
            - **Trazabilidad:** versión `{manifest["model_version"]}` con hashes verificados; la interfaz es de solo lectura y no reentrena nada.
            - **Alcance:** clasificación académica retrospectiva dentro del universo ONSV de siniestros fatales registrados. La disponibilidad de todos los campos al momento exacto de notificación no fue demostrada porque la fuente no incluye timestamps por variable.
            """
        )


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    priority = probability >= threshold
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44, "color": INK}},
            title={"text": "Probabilidad multifatal calibrada", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": ORANGE if priority else BLUE, "thickness": .42},
                "bgcolor": "#EEF0ED",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "#E7EFEE"},
                    {"range": [threshold * 100, 100], "color": "#F4E8DF"},
                ],
                "threshold": {"line": {"color": INK, "width": 4}, "thickness": .8, "value": threshold * 100},
            },
        )
    )
    return _plot_layout(fig, height=320)


def comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    data = comparison.dropna(subset=["tasa_multifatal_pct"]).copy()
    data["error_plus"] = data["ci_95_sup_pct"] - data["tasa_multifatal_pct"]
    data["error_minus"] = data["tasa_multifatal_pct"] - data["ci_95_inf_pct"]
    data.loc[data["comparador"] == "Probabilidad calibrada del modelo", ["error_plus", "error_minus"]] = 0
    colors = [ORANGE if value == "Probabilidad calibrada del modelo" else BLUE for value in data["comparador"]]
    symbols = ["diamond" if value == "Probabilidad calibrada del modelo" else "circle" for value in data["comparador"]]
    fig = go.Figure(
        go.Scatter(
            x=data["tasa_multifatal_pct"],
            y=data["comparador"],
            mode="markers",
            marker=dict(color=colors, symbol=symbols, size=13, line=dict(color="#FFFFFF", width=1.5)),
            error_x=dict(type="data", array=data["error_plus"], arrayminus=data["error_minus"], color=MUTED, thickness=1.4),
            customdata=np.column_stack([
                data["soporte"].fillna("—").astype(str), data["dimensiones"], data["fuente"]
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>Valor: %{x:.2f}%<br>n: %{customdata[0]}"
                "<br>%{customdata[1]}<br>Fuente: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Estimación y referencias observacionales")
    fig.update_xaxes(title="Probabilidad o tasa multifatal (%)", rangemode="tozero")
    fig.update_yaxes(title=None)
    return _plot_layout(fig, height=390)


PERU_BOUNDS = [[-18.6, -81.6], [0.3, -68.4]]


def _location_picker() -> None:
    """Optional OpenStreetMap picker: a click fills the latitude/longitude inputs.

    The numeric fields remain the canonical input (and what the tests and the
    geographic validation consume); the map is an additive convenience that
    degrades gracefully without the package or without tile connectivity.
    """
    try:
        import folium
        from streamlit_folium import st_folium
    except (ImportError, ModuleNotFoundError):
        st.info(
            "Para elegir la ubicación con un clic en el mapa, instalá las dependencias opcionales "
            "`folium` y `streamlit-folium` (incluidas en requirements). Mientras tanto podés escribir "
            "las coordenadas en el formulario."
        )
        return

    with st.expander("Elegir la ubicación en el mapa (OpenStreetMap)", expanded=False):
        st.caption(
            "Hacé clic sobre el punto del siniestro: la latitud y la longitud se completan solas en el "
            "formulario. El mapa muestra únicamente el Perú, con carreteras y calles de OpenStreetMap; "
            "requiere conexión a internet para cargar el fondo."
        )
        picked_latitude = st.session_state.get("input_latitude")
        picked_longitude = st.session_state.get("input_longitude")
        has_pick = picked_latitude is not None and picked_longitude is not None
        center = [picked_latitude, picked_longitude] if has_pick else [-9.19, -75.0]
        base_map = folium.Map(
            location=center,
            zoom_start=9 if has_pick else 5,
            min_zoom=5,
            max_bounds=True,
            tiles="OpenStreetMap",
            attr="© OpenStreetMap contributors",
        )
        if not has_pick:
            base_map.fit_bounds(PERU_BOUNDS)
        base_map.options["maxBounds"] = PERU_BOUNDS
        geo_path = ROOT / "data" / "geo" / "peru_departamentos_simple.geojson"
        if geo_path.exists():
            folium.GeoJson(
                json.loads(geo_path.read_text(encoding="utf-8")),
                name="Departamentos",
                style_function=lambda _: {"color": INK, "weight": 1, "fillOpacity": 0.02},
                tooltip=folium.GeoJsonTooltip(fields=["NOMBDEP"], aliases=["Departamento:"]),
            ).add_to(base_map)
        if has_pick:
            folium.Marker(
                [picked_latitude, picked_longitude],
                tooltip="Ubicación seleccionada",
                icon=folium.Icon(color="orange", icon="screenshot"),
            ).add_to(base_map)
        map_state = st_folium(
            base_map,
            key="location_picker_map",
            height=430,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )
        clicked = (map_state or {}).get("last_clicked")
        if clicked:
            clicked_latitude = round(float(clicked["lat"]), 6)
            clicked_longitude = round(float(clicked["lng"]), 6)
            if (clicked_latitude, clicked_longitude) != (picked_latitude, picked_longitude):
                st.session_state["input_latitude"] = clicked_latitude
                st.session_state["input_longitude"] = clicked_longitude
                matches = departments_for_point(clicked_latitude, clicked_longitude)
                if matches:
                    # Deterministic reverse lookup on the same versioned
                    # polygons used by input validation; a border click keeps
                    # the first match and validation re-checks on submit.
                    st.session_state["input_department"] = matches[0]
                st.session_state.pop("canonical_result", None)
                st.rerun()
        if has_pick:
            department_note = st.session_state.get("input_department")
            department_suffix = f" Departamento deducido: {department_note}." if department_note else ""
            st.caption(
                f"Seleccionado: {format_number_es(picked_latitude, digits=6)}, "
                f"{format_number_es(picked_longitude, digits=6)}.{department_suffix} "
                "El resto de los campos (zona, tipo de vía, clima…) describe cómo el ONSV registró el siniestro y no puede "
                "deducirse del mapa con honestidad; completalos según el registro. La validación Perú–departamento se aplica al enviar."
            )


def estimate_page() -> None:
    st.header("Probar la red neuronal")
    st.caption("Completá un registro o elegí uno de los 5 escenarios académicos. La salida es una probabilidad calibrada de multifatalidad y una clase estimada.")
    options = load_input_options()
    thresholds = load_thresholds()
    known_road_codes = load_known_road_codes()

    def option_label(value: str | None) -> str:
        return "NO INFORMADO" if value is None else value

    def optional_values(field: str) -> list[str | None]:
        return [None, *[value for value in options[field] if value is not None]]

    def number_default(key: str, default: int | float | None) -> int | float | None:
        """Avoid Streamlit's session-state/default conflict after loading a demo."""
        return None if key in st.session_state else default

    demos = load_demo_cases()
    demo_options = demos["caso_id"].tolist() if not demos.empty else []
    demo_labels = dict(zip(demos["caso_id"], demos["descripcion"])) if not demos.empty else {}
    demo_choice = st.selectbox(
        "Escenario de demostración",
        demo_options,
        format_func=lambda value: demo_labels.get(value, value),
        placeholder="Elegí uno de los 5 escenarios…",
        index=None,
        key="demo_scenario",
    )
    demo_clicked = st.button("Cargar escenario", key="load_demo_case", disabled=demo_choice is None)
    if demo_clicked:
        if demos.empty:
            st.error("No se encontró el caso de demostración canónico.")
        else:
            demo = demos.loc[demos["caso_id"].eq(demo_choice)].iloc[0]
            widget_values = {
                "input_date": pd.Timestamp(demo["FECHA"]).date(),
                "input_time": pd.Timestamp(str(demo["HORA"])).time(),
                "input_department": demo["DEPARTAMENTO"],
                "input_road_code": demo["CODIGO_VIA"],
                "input_class": demo["CLASE"],
                "input_zone": demo["ZONA"],
                "input_network": demo["RED_VIAL"],
                "input_road_type": demo["TIPO_VIA"],
                "input_weather": demo["CLIMA"],
                "input_characteristic": demo["CARACTERISTICA_VIA"],
                "input_profile": demo["PERFIL_VIA"],
                "input_surface": demo["SUPERFICIE"],
                "input_latitude": float(demo["LATITUD"]),
                "input_longitude": float(demo["LONGITUD"]),
                "input_n_vehiculos": int(demo["n_vehiculos"]),
                "input_n_bus": int(demo["n_bus"]),
                "input_n_pesado_carga": int(demo["n_pesado_carga"]),
                "input_n_moto": int(demo["n_moto"]),
                "input_n_no_identificado": int(demo["n_no_identificado"]),
                "input_n_interprovincial": int(demo["n_interprovincial"]),
                "input_n_transporte_publico": int(demo["n_transporte_publico"]),
                "input_n_personas": int(demo["n_personas"]),
                "input_n_pasajeros": int(demo["n_pasajeros"]),
                "input_n_peatones": int(demo["n_peatones"]),
                "input_n_conductor_fugado": int(demo["n_conductor_fugado"]),
                "input_edad_media": float(demo["edad_media_involucrados"]) if pd.notna(demo["edad_media_involucrados"]) else None,
            }
            st.session_state.update(widget_values)
            st.session_state.pop("canonical_result", None)
            st.toast("Escenario cargado. Modificá un campo por vez para observar la respuesta del modelo; el cambio no implica causalidad.")

    st.subheader("Ubicación")
    _location_picker()

    with st.form("canonical_estimate_form", clear_on_submit=False):
        st.subheader("Evento y ubicación")
        top = st.columns(4)
        with top[0]:
            incident_date = st.date_input(
                "Fecha del siniestro",
                value=None,
                min_value=INFERENCE_DATE_MIN,
                max_value=INFERENCE_DATE_MAX,
                format="DD/MM/YYYY",
                key="input_date",
                help=f"Periodo académico: {format_date_es(INFERENCE_DATE_MIN)}–{format_date_es(INFERENCE_DATE_MAX)}.",
            )
        with top[1]:
            incident_time = st.time_input("Hora del siniestro", value=None, step=300, key="input_time")
        with top[2]:
            department = st.selectbox(
                "Departamento",
                options["DEPARTAMENTO"],
                index=None,
                placeholder="Seleccioná un departamento…",
                format_func=option_label,
                key="input_department",
            )
        with top[3]:
            road_code_options: list[str | None] = [None, *known_road_codes]
            loaded_road_code = normalize_road_code(st.session_state.get("input_road_code"))
            if loaded_road_code != "DESCONOCIDO" and loaded_road_code not in known_road_codes:
                # Streamlit validates session state against widget options even
                # with accept_new_options=True. Preserve an unseen demo/user
                # value explicitly so it renders and reaches the canonical
                # frequency-zero path instead of raising ValueError.
                road_code_options.append(loaded_road_code)
            road_code = st.selectbox(
                "Código de vía (opcional)",
                road_code_options,
                index=0,
                placeholder="Buscá o escribí un código…",
                format_func=option_label,
                accept_new_options=True,
                key="input_road_code",
                help="Acepta códigos del entrenamiento y nuevos códigos con formato oficial, por ejemplo PE-1N.",
            )
            normalized_road_code = normalize_road_code(road_code)
            if normalized_road_code != "DESCONOCIDO" and normalized_road_code not in set(known_road_codes):
                st.warning("Código no observado en entrenamiento: el modelo conservará el mapeo canónico con frecuencia 0.")

        st.markdown("#### Vía y entorno")
        middle = st.columns(4)
        with middle[0]:
            crash_class = st.selectbox("Clase de siniestro", options["CLASE"], index=None, placeholder="Seleccioná una clase…", format_func=option_label, key="input_class")
        with middle[1]:
            zone = st.selectbox("Zona", optional_values("ZONA"), index=0, format_func=option_label, key="input_zone")
        with middle[2]:
            road_network = st.selectbox("Red vial", optional_values("RED_VIAL"), index=0, format_func=option_label, key="input_network")
        with middle[3]:
            road_type = st.selectbox("Tipo de vía", optional_values("TIPO_VIA"), index=0, format_func=option_label, key="input_road_type")

        bottom = st.columns(4)
        with bottom[0]:
            weather = st.selectbox("Clima", optional_values("CLIMA"), index=0, format_func=option_label, key="input_weather")
        with bottom[1]:
            characteristic = st.selectbox("Característica vial", optional_values("CARACTERISTICA_VIA"), index=0, format_func=option_label, key="input_characteristic")
        with bottom[2]:
            profile = st.selectbox("Perfil longitudinal", optional_values("PERFIL_VIA"), index=0, format_func=option_label, key="input_profile")
        with bottom[3]:
            surface = st.selectbox("Superficie", optional_values("SUPERFICIE"), index=0, format_func=option_label, key="input_surface")

        st.markdown("#### Coordenadas (requeridas)")
        coordinate_columns = st.columns(2)
        with coordinate_columns[0]:
            latitude = st.number_input("Latitud", min_value=-90.0, max_value=90.0, value=None, format="%.6f", placeholder="Ej.: -12.046374…", key="input_latitude")
        with coordinate_columns[1]:
            longitude = st.number_input("Longitud", min_value=-180.0, max_value=180.0, value=None, format="%.6f", placeholder="Ej.: -77.042793…", key="input_longitude")
        st.caption("Ingresá ambas coordenadas. El esquema canónico exige ubicación para la inferencia final; la app no imputa una localización silenciosamente.")

        st.markdown("#### Vehículos y personas")
        scene_top = st.columns(4)
        with scene_top[0]:
            n_vehiculos = st.number_input("Vehículos involucrados", min_value=1, max_value=10, value=None, step=1, placeholder="Ej.: 2", key="input_n_vehiculos")
        with scene_top[1]:
            n_personas = st.number_input("Personas involucradas", min_value=1, max_value=120, value=None, step=1, placeholder="Ej.: 4", key="input_n_personas")
        with scene_top[2]:
            n_pasajeros = st.number_input("De ellas, pasajeros/ocupantes", min_value=0, max_value=120, value=number_default("input_n_pasajeros", 0), step=1, key="input_n_pasajeros")
        with scene_top[3]:
            n_peatones = st.number_input("De ellas, peatones", min_value=0, max_value=120, value=number_default("input_n_peatones", 0), step=1, key="input_n_peatones")
        scene_bottom = st.columns(4)
        with scene_bottom[0]:
            n_bus = st.number_input("Buses / minibuses", min_value=0, max_value=10, value=number_default("input_n_bus", 0), step=1, key="input_n_bus")
        with scene_bottom[1]:
            n_pesado_carga = st.number_input("Pesados de carga", min_value=0, max_value=10, value=number_default("input_n_pesado_carga", 0), step=1, key="input_n_pesado_carga")
        with scene_bottom[2]:
            n_moto = st.number_input("Motos / trimotos / bicis", min_value=0, max_value=10, value=number_default("input_n_moto", 0), step=1, key="input_n_moto")
        with scene_bottom[3]:
            n_no_identificado = st.number_input("Vehículos no identificados", min_value=0, max_value=10, value=number_default("input_n_no_identificado", 0), step=1, key="input_n_no_identificado")
        scene_extra = st.columns(4)
        with scene_extra[0]:
            n_interprovincial = st.number_input("Servicio interprovincial", min_value=0, max_value=10, value=number_default("input_n_interprovincial", 0), step=1, key="input_n_interprovincial")
        with scene_extra[1]:
            n_transporte_publico = st.number_input("Transporte público / taxi", min_value=0, max_value=10, value=number_default("input_n_transporte_publico", 0), step=1, key="input_n_transporte_publico")
        with scene_extra[2]:
            n_conductor_fugado = st.number_input("Conductores fugados", min_value=0, max_value=10, value=number_default("input_n_conductor_fugado", 0), step=1, key="input_n_conductor_fugado")
        with scene_extra[3]:
            edad_media = st.number_input("Edad media involucrados (opcional)", min_value=0.0, max_value=110.0, value=None, step=1.0, placeholder="NO INFORMADO", key="input_edad_media")
        st.caption(
            "Conteos registrados en las tablas complementarias. La fuente no permite probar que todos estuvieran disponibles en el instante de la notificación; "
            "por eso la demostración se interpreta retrospectivamente. No se ingresa ningún desenlace por persona."
        )
        submitted = st.form_submit_button("Calcular clase estimada", type="primary")

    if submitted:
        st.session_state.pop("canonical_result", None)
        st.session_state.pop("prediction_result", None)
        record = pd.DataFrame(
            [{
                "FECHA": incident_date.isoformat() if incident_date is not None else None,
                "HORA": incident_time.strftime("%H:%M") if incident_time is not None else None,
                "DEPARTAMENTO": department,
                "CODIGO_VIA": normalize_road_code(road_code),
                "LATITUD": latitude,
                "LONGITUD": longitude,
                "CLASE": crash_class,
                "ZONA": zone,
                "RED_VIAL": road_network,
                "TIPO_VIA": road_type,
                "CLIMA": weather,
                "CARACTERISTICA_VIA": characteristic,
                "PERFIL_VIA": profile,
                "SUPERFICIE": surface,
                "n_vehiculos": n_vehiculos,
                "n_bus": n_bus,
                "n_pesado_carga": n_pesado_carga,
                "n_moto": n_moto,
                "n_no_identificado": n_no_identificado,
                "n_interprovincial": n_interprovincial,
                "n_transporte_publico": n_transporte_publico,
                "n_personas": n_personas,
                "n_pasajeros": n_pasajeros,
                "n_peatones": n_peatones,
                "n_conductor_fugado": n_conductor_fugado,
                "edad_media_involucrados": edad_media,
            }]
        )
        try:
            with st.spinner("Verificando contrato y ejecutando el modelo…"):
                prediction = predict_records(record).iloc[0]
                comparison = historical_comparison(record.iloc[0], float(prediction["calibrated_probability"]))
            st.session_state["canonical_result"] = {"prediction": prediction.to_dict(), "comparison": comparison}
        except InputContractError as exc:
            st.error(f"Revisá el formulario: {exc}")
        except RuntimeArtifactError as exc:
            st.error(str(exc))

    result = st.session_state.get("canonical_result")
    if not result:
        st.info("Todavía no hay una estimación. Enviá el formulario para ver un resultado estable y trazable.")
        return

    prediction = result["prediction"]
    comparison = result["comparison"]
    calibrated_probability = float(prediction["calibrated_probability"])
    calibrated_threshold = float(prediction["calibrated_threshold"])
    multifatal_class = calibrated_probability >= calibrated_threshold

    st.subheader("Resultado")
    left, right = st.columns([1.15, .85])
    with left:
        st.plotly_chart(probability_gauge(calibrated_probability, calibrated_threshold), width="stretch", config={"displayModeBar": False})
        st.caption(f"La línea marca el umbral calibrado de {_percent(calibrated_threshold, 0)}. Esta escala es la única usada para la decisión visible.")
    with right:
        symbol = "2+" if multifatal_class else "1"
        estimated_class = "2+ fallecidos" if multifatal_class else "1 fallecido"
        st.markdown(
            f'<div class="decision-panel"><div class="evidence-label">Clase estimada</div>'
            f'<div class="decision-title">{symbol} · {estimated_class}</div>'
            '<div class="decision-copy">Clasificación académica de multifatalidad obtenida con Platt y el umbral '
            'seleccionado exclusivamente en validación 2023. No es una explicación causal del caso.</div></div>',
            unsafe_allow_html=True,
        )
    st.subheader("Contexto histórico")
    st.plotly_chart(comparison_chart(comparison), width="stretch", config={"displayModeBar": False})
    st.caption("Los intervalos son Wilson 95%. El subgrupo coincidente se informa cuando alcanza soporte n ≥ 30.")
    display = comparison[["comparador", "tasa_multifatal_pct", "soporte", "ci_95_inf_pct", "ci_95_sup_pct", "dimensiones", "fuente"]].copy()
    display.columns = ["Comparador", "Valor (%)", "n", "IC 95% inf. (%)", "IC 95% sup. (%)", "Dimensiones", "Fuente"]
    _table_fallback("contexto de la estimación", display.round(2), key="contexto_estimacion")

    with st.expander("Diagnóstico técnico: escala cruda separada"):
        st.write(
            "La salida sigmoide cruda y su umbral pertenecen a otra escala. Se muestran solo para auditoría técnica; "
            "no se comparan ni se superponen con la probabilidad calibrada."
        )
        diagnostics = pd.DataFrame(
            [
                {"Escala": "Calibrada (clase visible)", "Score": calibrated_probability, "Umbral": calibrated_threshold, "Clase": int(calibrated_probability >= calibrated_threshold)},
                {"Escala": "Cruda (diagnóstico)", "Score": float(prediction["raw_probability"]), "Umbral": float(prediction["raw_threshold"]), "Clase": int(prediction["raw_prediction"])},
            ]
        )
        st.dataframe(diagnostics.style.format({"Score": "{:.4f}", "Umbral": "{:.2f}"}), width="stretch", hide_index=True)
        st.caption(f'Método de calibración persistido: {prediction["calibration_method"]}. Fuente de umbrales: models/final/thresholds.json.')


def _aggregate_rate(df: pd.DataFrame, column: str, dimension: str) -> pd.DataFrame:
    rows = []
    for label, subset in df.groupby(column, dropna=False):
        n = int(len(subset))
        positives = int(subset["target_multifatal"].sum())
        lower, upper = wilson_interval(positives, n)
        rows.append({"dimension": dimension, "categoria": str(label), "n": n, "multifatales": positives, "tasa": positives / n, "ci_inf": lower, "ci_sup": upper})
    return pd.DataFrame(rows)


def explore_page() -> None:
    st.header("Explorar datos")
    st.caption("Gráficos responsivos calculados desde el parquet canónico; no se usan capturas EDA estáticas.")
    df = cached_clean_dataset()
    df["FECHA"] = pd.to_datetime(df["FECHA"])

    st.subheader("1. Balance del target")
    target = (
        df["target_multifatal"].value_counts().rename_axis("target").reset_index(name="n").sort_values("target")
    )
    target["clase"] = target["target"].map({0: "1 fallecido", 1: "2+ fallecidos"})
    target["porcentaje"] = target["n"] / len(df) * 100
    target_fig = go.Figure(go.Bar(
        x=target["clase"], y=target["n"], marker_color=[BLUE_LIGHT, ORANGE],
        text=[f'{format_number_es(n)}<br>{format_number_es(p, digits=1, suffix=" %")}' for n, p in zip(target["n"], target["porcentaje"])], textposition="outside",
        hovertemplate="%{x}<br>n=%{y:,}<extra></extra>",
    ))
    target_fig.update_layout(title="Distribución de siniestros fatales por clase")
    target_fig.update_xaxes(title=None)
    target_fig.update_yaxes(title="Siniestros fatales", rangemode="tozero")
    st.plotly_chart(_plot_layout(target_fig, height=390), width="stretch", config={"displayModeBar": False})
    _table_fallback("distribución del target", target[["clase", "n", "porcentaje"]].round(2), key="target")

    st.subheader("2. Volumen mensual")
    monthly = df.set_index("FECHA").resample("MS").size().rename("siniestros_fatales").reset_index()
    monthly_fig = go.Figure(go.Scatter(
        x=monthly["FECHA"], y=monthly["siniestros_fatales"], mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=5, color=BLUE),
        hovertemplate="%{x|%b %Y}<br>Siniestros: %{y:,}<extra></extra>",
    ))
    monthly_fig.add_vrect(x0="2025-01-01", x1="2025-12-31", fillcolor=ORANGE, opacity=.08, line_width=0,
                          annotation_text="2025 preliminar", annotation_position="top left")
    monthly_fig.update_layout(title="Siniestros fatales registrados por mes")
    monthly_fig.update_xaxes(title="Mes")
    monthly_fig.update_yaxes(title="Siniestros fatales", rangemode="tozero")
    st.plotly_chart(_plot_layout(monthly_fig), width="stretch", config={"displayModeBar": False})
    _table_fallback("volumen mensual", monthly.assign(FECHA=monthly["FECHA"].dt.strftime("%Y-%m")), key="volumen_mensual")

    st.subheader("3. Tasa por clase de siniestro")
    class_rate = _aggregate_rate(df, "CLASE", "Clase").sort_values("tasa")
    class_rate["tasa_pct"] = class_rate["tasa"] * 100
    class_rate["ci_inf_pct"] = class_rate["ci_inf"] * 100
    class_rate["ci_sup_pct"] = class_rate["ci_sup"] * 100
    class_fig = go.Figure(go.Scatter(
        x=class_rate["tasa_pct"], y=class_rate["categoria"], mode="markers+text",
        marker=dict(color=ORANGE, size=12, symbol="diamond"),
        text=[f"n={format_number_es(n)}" for n in class_rate["n"]], textposition="middle right",
        error_x=dict(type="data", array=class_rate["ci_sup_pct"] - class_rate["tasa_pct"], arrayminus=class_rate["tasa_pct"] - class_rate["ci_inf_pct"], color=MUTED),
        customdata=class_rate[["n", "multifatales", "ci_inf_pct", "ci_sup_pct"]],
        hovertemplate="<b>%{y}</b><br>Tasa: %{x:.2f}%<br>n=%{customdata[0]:,}<br>Multifatales=%{customdata[1]:,}<br>IC 95%%: %{customdata[2]:.2f}–%{customdata[3]:.2f}%<extra></extra>",
    ))
    class_fig.update_layout(title="Tasa multifatal por clase, con IC Wilson 95%")
    class_fig.update_xaxes(title="Tasa multifatal (%)", rangemode="tozero")
    class_fig.update_yaxes(title=None)
    st.plotly_chart(_plot_layout(class_fig), width="stretch", config={"displayModeBar": False})
    _table_fallback("tasa por clase", class_rate[["categoria", "n", "multifatales", "tasa_pct", "ci_inf_pct", "ci_sup_pct"]].round(2), key="tasa_clase")

    st.subheader("4. Condiciones registradas")
    condition_specs = [("ZONA", "Zona"), ("CLIMA", "Clima"), ("CARACTERISTICA_VIA", "Geometría"), ("SUPERFICIE", "Superficie")]
    conditions = pd.concat([_aggregate_rate(df, column, label) for column, label in condition_specs], ignore_index=True)
    conditions["tasa_pct"] = conditions["tasa"] * 100
    conditions["ci_inf_pct"] = conditions["ci_inf"] * 100
    conditions["ci_sup_pct"] = conditions["ci_sup"] * 100
    max_rate = max(1.0, float(conditions["ci_sup_pct"].max()) * 1.12)
    facets = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in condition_specs], horizontal_spacing=.12, vertical_spacing=.2)
    for position, (_, label) in enumerate(condition_specs):
        subset = conditions[conditions["dimension"] == label].sort_values("tasa_pct")
        row, column = divmod(position, 2)
        facets.add_trace(go.Scatter(
            x=subset["tasa_pct"], y=subset["categoria"], mode="markers",
            marker=dict(color=BLUE, size=9),
            error_x=dict(type="data", array=subset["ci_sup_pct"] - subset["tasa_pct"], arrayminus=subset["tasa_pct"] - subset["ci_inf_pct"], color=MUTED),
            customdata=subset[["n", "ci_inf_pct", "ci_sup_pct"]],
            hovertemplate="<b>%{y}</b><br>Tasa: %{x:.2f}%<br>n=%{customdata[0]:,}<br>IC 95%%: %{customdata[1]:.2f}–%{customdata[2]:.2f}%<extra></extra>",
            showlegend=False,
        ), row=row + 1, col=column + 1)
        facets.update_xaxes(range=[0, max_rate], title_text="Tasa multifatal (%)", row=row + 1, col=column + 1)
    facets.update_layout(title="Tasas descriptivas por condiciones; misma escala en todos los paneles")
    st.plotly_chart(_plot_layout(facets, height=700), width="stretch", config={"displayModeBar": False})
    _table_fallback("tasas por condiciones", conditions[["dimension", "categoria", "n", "multifatales", "tasa_pct", "ci_inf_pct", "ci_sup_pct"]].round(2), key="condiciones")

    st.subheader("5. Mes × hora")
    temporal = df.dropna(subset=["hora_entera"]).copy()
    temporal["mes"] = temporal["FECHA"].dt.month
    heat = temporal.pivot_table(index="hora_entera", columns="mes", values="target_multifatal", aggfunc="size", fill_value=0)
    heat = heat.reindex(index=range(24), columns=range(1, 13), fill_value=0)
    heat_fig = go.Figure(go.Heatmap(
        z=heat.to_numpy(), x=["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
        y=[f"{hour:02d}:00" for hour in heat.index], colorscale=[[0, "#EEF2F1"], [1, BLUE]],
        colorbar=dict(title="n"), hovertemplate="Mes: %{x}<br>Hora: %{y}<br>Siniestros: %{z:,}<extra></extra>",
    ))
    heat_fig.update_layout(title="Volumen observado de siniestros fatales por mes y hora")
    heat_fig.update_xaxes(title="Mes")
    heat_fig.update_yaxes(title="Hora")
    st.plotly_chart(_plot_layout(heat_fig, height=610), width="stretch", config={"displayModeBar": False})
    month_names = {month: name for month, name in enumerate(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], start=1)}
    heat_table = heat.rename(columns=month_names).reset_index().rename(columns={"hora_entera": "hora"})
    _table_fallback("matriz mes por hora", heat_table, key="mes_hora")

    st.subheader("6. Cobertura geográfica")
    geo = df.dropna(subset=["LATITUD", "LONGITUD"]).copy()
    geo["clase_target"] = geo["target_multifatal"].map({0: "1 fallecido", 1: "2+ fallecidos"})
    geo_fig = go.Figure()
    for target_class, color, symbol in [("1 fallecido", BLUE, "circle"), ("2+ fallecidos", ORANGE, "diamond")]:
        subset = geo[geo["clase_target"] == target_class]
        geo_fig.add_trace(go.Scattergeo(
            lon=subset["LONGITUD"], lat=subset["LATITUD"], mode="markers", name=target_class,
            marker=dict(size=5 if target_class == "1 fallecido" else 7, color=color, symbol=symbol, opacity=.48 if target_class == "1 fallecido" else .78),
            customdata=subset[["DEPARTAMENTO", "FECHA"]].astype(str),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Clase: " + target_class + "<extra></extra>",
        ))
    geo_fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    geo_fig.update_layout(title="Registros con coordenadas válidas", legend_title_text="Clase observada")
    st.plotly_chart(_plot_layout(geo_fig, height=590, legend=True), width="stretch")
    st.caption(f"Se muestran {format_number_es(len(geo))} de {format_number_es(len(df))} registros con ambas coordenadas. El símbolo, además del color, distingue la clase.")
    _table_fallback("cobertura geográfica por departamento", geo.groupby(["DEPARTAMENTO", "clase_target"]).size().reset_index(name="n"), key="cobertura_geo")


def regional_map(regional: pd.DataFrame, geojson: dict[str, Any]) -> go.Figure:
    supported = regional[regional["soporte_suficiente"]].copy()
    unsupported = regional[~regional["soporte_suficiente"]].copy()
    fig = go.Figure()
    if not unsupported.empty:
        fig.add_trace(go.Choropleth(
            geojson=geojson, locations=unsupported["DEPARTAMENTO"], featureidkey="properties.NOMBDEP",
            z=np.zeros(len(unsupported)), colorscale=[[0, UNSUPPORTED], [1, UNSUPPORTED]], showscale=False,
            customdata=unsupported[["siniestros_fatales"]],
            hovertemplate="<b>%{location}</b><br>n=%{customdata[0]:,}<br>Sin tasa: soporte < 30<extra></extra>",
            name="Soporte insuficiente", marker_line_color="#FFFFFF", marker_line_width=.7,
        ))
    if not supported.empty:
        custom = np.column_stack([
            supported["siniestros_fatales"], supported["multifatales"],
            supported["ci_95_inf"] * 100, supported["ci_95_sup"] * 100,
        ])
        fig.add_trace(go.Choropleth(
            geojson=geojson, locations=supported["DEPARTAMENTO"], featureidkey="properties.NOMBDEP",
            z=supported["tasa_multifatal"] * 100, colorscale=[[0, "#E8EFEE"], [1, BLUE]],
            colorbar=dict(title="Tasa (%)"),
            customdata=custom,
            hovertemplate="<b>%{location}</b><br>Tasa: %{z:.2f}%<br>n=%{customdata[0]:,}<br>Multifatales=%{customdata[1]:,}<br>IC 95%%: %{customdata[2]:.2f}–%{customdata[3]:.2f}%<extra></extra>",
            name="Soporte suficiente", marker_line_color="#FFFFFF", marker_line_width=.7,
        ))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=f"Tasa multifatal por departamento (se informa con n ≥ {MINIMUM_REGIONAL_SUPPORT})")
    return _plot_layout(fig, height=620, legend=True)


def regional_page() -> None:
    st.header("Patrones regionales")
    st.write(
        f"La tasa se informa solo con soporte n ≥ {MINIMUM_REGIONAL_SUPPORT}. Las regiones por debajo del mínimo quedan en gris; "
        "no se las convierte en cero ni se las compara como si fueran estimaciones estables."
    )
    regional = cached_regional_summary()
    geo_path = ROOT / "data" / "geo" / "peru_departamentos_simple.geojson"
    if geo_path.exists():
        try:
            geojson = json.loads(geo_path.read_text(encoding="utf-8"))
            st.plotly_chart(regional_map(regional, geojson), width="stretch", config={"displayModeBar": False})
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            st.warning(f"No se pudo renderizar el mapa: {exc}. El ranking tabular permanece disponible.")
    else:
        st.info("El GeoJSON departamental no está disponible. El ranking tabular permanece disponible.")

    supported = regional[regional["soporte_suficiente"]].copy().sort_values("tasa_multifatal", ascending=False)
    supported["tasa_multifatal_pct"] = supported["tasa_multifatal"] * 100
    supported["ci_95_inf_pct"] = supported["ci_95_inf"] * 100
    supported["ci_95_sup_pct"] = supported["ci_95_sup"] * 100
    ranking_fig = go.Figure(go.Bar(
        x=supported["tasa_multifatal_pct"], y=supported["DEPARTAMENTO"], orientation="h",
        marker_color=BLUE, text=[f"{format_number_es(value, digits=1, suffix=' %')} · n={format_number_es(n)}" for value, n in zip(supported["tasa_multifatal_pct"], supported["siniestros_fatales"])],
        textposition="outside",
        error_x=dict(type="data", array=supported["ci_95_sup_pct"] - supported["tasa_multifatal_pct"], arrayminus=supported["tasa_multifatal_pct"] - supported["ci_95_inf_pct"], color=MUTED),
        customdata=supported[["siniestros_fatales", "multifatales", "ci_95_inf_pct", "ci_95_sup_pct"]],
        hovertemplate="<b>%{y}</b><br>Tasa: %{x:.2f}%<br>n=%{customdata[0]:,}<br>Multifatales=%{customdata[1]:,}<br>IC 95%%: %{customdata[2]:.2f}–%{customdata[3]:.2f}%<extra></extra>",
    ))
    ranking_fig.update_layout(title="Ranking departamental con incertidumbre")
    ranking_fig.update_xaxes(title="Tasa multifatal (%)", rangemode="tozero")
    ranking_fig.update_yaxes(title=None, autorange="reversed")
    st.plotly_chart(_plot_layout(ranking_fig, height=760), width="stretch", config={"displayModeBar": False})

    table = mask_unsupported_regional_rates(regional, MINIMUM_REGIONAL_SUPPORT)
    for column in ("tasa_multifatal", "ci_95_inf", "ci_95_sup"):
        table[column] = table[column] * 100
    table.columns = ["Departamento", "Siniestros fatales", "Multifatales", "Tasa (%)", "IC 95% inf. (%)", "IC 95% sup. (%)", "Soporte suficiente"]
    st.subheader("Tabla regional completa")
    st.dataframe(table.round(2), width="stretch", hide_index=True)
    st.download_button(
        "Descargar tabla regional",
        table.to_csv(index=False).encode("utf-8"),
        file_name="patrones_regionales_onsv.csv",
        mime="text/csv",
    )


def _ci_lookup(ci: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    row = ci[(ci["probability_scale"] == "calibrated") & (ci["metric"] == metric)].iloc[0]
    return float(row["estimate"]), float(row["ci_2_5"]), float(row["ci_97_5"])


def _model_comparison_table(baseline: pd.DataFrame, calibration_method: str) -> pd.DataFrame:
    names = {
        "MLP_definitiva": "MLP canónica",
        "LogisticRegression_balanced": "Regresión logística",
        "RandomForest_balanced": "Random Forest",
    }
    comparison = baseline[
        ((baseline["model"] == "MLP_definitiva") & (baseline["probability_scale"] == calibration_method))
        | baseline["model"].isin(["LogisticRegression_balanced", "RandomForest_balanced"])
    ].copy()
    comparison["Modelo"] = comparison["model"].map(names)
    if len(comparison) != 3 or comparison["Modelo"].isna().any() or comparison["Modelo"].nunique() != 3:
        raise RuntimeArtifactError("La comparación canónica debe contener exactamente la MLP, la regresión logística y el Random Forest.")
    comparison["F1"] = comparison["f1_multifatal"]
    comparison["PR-AUC"] = comparison["pr_auc"]
    comparison["ROC-AUC"] = comparison["roc_auc"]
    return comparison


def _model_leadership_text(comparison: pd.DataFrame) -> str:
    leaders = [
        f"{metric}: {comparison.loc[comparison[metric].idxmax(), 'Modelo']} ({comparison[metric].max():.4f})"
        for metric in ("PR-AUC", "ROC-AUC", "F1")
    ]
    return "Liderazgos nominales en esta referencia: " + "; ".join(leaders) + ". No se afirma superioridad universal."


def evidence_page(manifest: dict[str, Any]) -> None:
    st.header("Evidencia del modelo")
    evidence = load_reference_artifacts()
    metrics = evidence["metrics"]["calibrated"]
    ci = evidence["ci"]

    st.warning(
        "Las métricas 2024–2025 son referencia histórica: sus etiquetas ya fueron observadas y no autorizan nuevos ajustes. "
        "No se lo presenta como un test nuevo ni se lo usa para reajustar el modelo."
    )
    columns = st.columns(4)
    metric_specs = [
        ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("F1 multifatal", "f1_multifatal"), ("ECE · 10 bins", "ece_10_bins"),
    ]
    for column, (label, key) in zip(columns, metric_specs):
        estimate, lower, upper = _ci_lookup(ci, key)
        with column:
            _card(label, format_number_es(estimate, digits=3), f"IC bootstrap 95 %: {format_number_es(lower, digits=3)}–{format_number_es(upper, digits=3)}")
    st.caption(f'Prevalencia de la clase positiva: {_percent(metrics["class_rate"])}. PR-AUC debe leerse contra esta referencia, no contra 0.50.')

    st.subheader("¿La red necesita más regularización o más de una red?")
    design = load_design_artifacts()
    regularization = design["regularization"].copy()  # type: ignore[union-attr]
    strategies = design["strategies"].copy()  # type: ignore[union-attr]
    strategy_ci = design["strategy_bootstrap"].copy()  # type: ignore[union-attr]
    audit = design["audit"]  # type: ignore[assignment]
    strategy_display = strategy_presentation_table(strategies)

    reg_fig = go.Figure(go.Bar(
        x=regularization["median_pr_auc"], y=regularization["audit_id"], orientation="h",
        marker_color=[ORANGE, BLUE, "#7E9187", "#B6AAA0"],
        error_x=dict(type="data", array=regularization["iqr_pr_auc"] / 2, color=MUTED),
        customdata=regularization[["median_roc_auc", "median_f1_multifatal", "seeds"]],
        hovertemplate="<b>%{y}</b><br>PR-AUC mediana: %{x:.4f}<br>ROC-AUC mediana: %{customdata[0]:.4f}<br>F1 mediana: %{customdata[1]:.4f}<br>Semillas: %{customdata[2]:.0f}<extra></extra>",
    ))
    reg_fig.update_layout(title="Ablación validation-only: L2 y dropout")
    reg_fig.update_xaxes(title="PR-AUC mediana en 2023", range=[0, max(.55, float(regularization["median_pr_auc"].max()) * 1.12)])
    reg_fig.update_yaxes(title=None, autorange="reversed")
    st.plotly_chart(_plot_layout(reg_fig, height=390), width="stretch", config={"displayModeBar": False})
    st.caption(
        "La configuración canónica ofrece el mejor compromiso de ROC-AUC mediana y baja dispersión, pero la ablación no demuestra que cada regularizador sea indispensable por separado: "
        "solo dropout tiene una PR-AUC mediana apenas mayor. Se conserva la configuración predeclarada; no se ajusta el modelo con esta auditoría posterior."
    )

    strategy_fig = go.Figure(go.Bar(
        x=strategy_display["label"],
        y=strategy_display["pr_auc"], marker_color=strategy_display["color"],
        text=[format_number_es(value, digits=3) for value in strategy_display["pr_auc"]], textposition="outside",
        customdata=strategy_display[["roc_auc", "f1_multifatal", "threshold_validation"]],
        hovertemplate="<b>%{x}</b><br>PR-AUC: %{y:.4f}<br>ROC-AUC: %{customdata[0]:.4f}<br>F1: %{customdata[1]:.4f}<br>Umbral 2023: %{customdata[2]:.2f}<extra></extra>",
    ))
    strategy_fig.update_layout(title="Una red frente a alternativas más complejas · validación 2023")
    strategy_fig.update_yaxes(title="PR-AUC", range=[0, .58])
    st.plotly_chart(_plot_layout(strategy_fig, height=390), width="stretch", config={"displayModeBar": False})

    ci_order = strategy_ci.set_index("metric").loc[["pr_auc", "roc_auc", "f1_multifatal"]].reset_index()
    forest = go.Figure(go.Scatter(
        x=ci_order["delta"], y=["PR-AUC", "ROC-AUC", "F1"], mode="markers",
        marker=dict(color=ORANGE, size=11, symbol="diamond"),
        error_x=dict(type="data", array=ci_order["ci_97_5"] - ci_order["delta"], arrayminus=ci_order["delta"] - ci_order["ci_2_5"], color=BLUE, thickness=2, width=5),
        customdata=ci_order[["ci_2_5", "ci_97_5"]],
        hovertemplate="%{y}<br>Δ ensemble−1 red: %{x:+.4f}<br>IC 95%%: [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<extra></extra>",
    ))
    forest.add_vline(x=0, line_color=INK, line_width=1)
    forest.update_layout(title="Bootstrap pareado: ensemble − MLP congelada")
    forest.update_xaxes(title="Diferencia en validación 2023")
    forest.update_yaxes(title=None)
    st.plotly_chart(_plot_layout(forest, height=330), width="stretch", config={"displayModeBar": False})
    ci_zero = strategy_ci_zero_summary(strategy_ci)
    st.info(
        f"{ci_zero['copy']}. El ensemble mejora nominalmente, pero no hay evidencia suficiente para sustituir la única MLP canónica. "
        "Sería una hipótesis futura que exige un nuevo holdout y calibración independiente."
    )
    _table_fallback("ablación de regularización", regularization.round(5), key="ablacion_regularizacion")
    _table_fallback("estrategias de una o varias redes", strategies.round(5), key="estrategias_redes")

    st.subheader("Comparación honesta con baselines")
    baseline = evidence["baseline"].copy()
    comparison = _model_comparison_table(baseline, str(manifest["calibration"]["method"]))
    model_fig = go.Figure()
    for metric, color, symbol in [("F1", ORANGE, "diamond"), ("PR-AUC", BLUE, "circle"), ("ROC-AUC", "#7A8A91", "square")]:
        model_fig.add_trace(go.Scatter(
            x=comparison[metric], y=comparison["Modelo"], mode="markers", name=metric,
            marker=dict(color=color, symbol=symbol, size=12),
            hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.3f}}<extra></extra>",
        ))
    model_fig.update_layout(title="Métricas en la referencia temporal fija", legend_title_text="Métrica")
    model_fig.update_xaxes(title="Valor", range=[0, 1])
    model_fig.update_yaxes(title=None)
    st.plotly_chart(_plot_layout(model_fig, height=390, legend=True), width="stretch", config={"displayModeBar": False})
    st.info(_model_leadership_text(comparison))
    _table_fallback("comparación de modelos", comparison[["Modelo", "F1", "PR-AUC", "ROC-AUC", "precision_multifatal", "recall_multifatal", "threshold"]].round(4), key="modelos")

    st.subheader("¿La MLP aporta más que contar personas?")
    persons = design["persons"].copy()  # type: ignore[union-attr]
    persons_ci = design["persons_bootstrap"].copy()  # type: ignore[union-attr]
    person_names = person_strategy_labels(persons, audit)
    persons["Modelo"] = persons["model"].map(person_names)
    persons_fig = go.Figure()
    for metric, color, symbol in (("pr_auc", ORANGE, "diamond"), ("roc_auc", BLUE, "circle"), ("f1_multifatal", "#7E9187", "square")):
        persons_fig.add_trace(go.Scatter(
            x=persons[metric], y=persons["Modelo"], mode="markers", name=metric.replace("_multifatal", "").upper(),
            marker=dict(color=color, symbol=symbol, size=12),
            hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>",
        ))
    persons_fig.update_layout(title="Referencia histórica post-hoc · umbrales elegidos solo en 2023")
    persons_fig.update_xaxes(title="Valor", range=[0, 1])
    persons_fig.update_yaxes(title=None)
    st.plotly_chart(_plot_layout(persons_fig, height=330, legend=True), width="stretch", config={"displayModeBar": False})
    pr_delta = persons_ci.loc[persons_ci["metric"].eq("pr_auc")].iloc[0]
    roc_delta = persons_ci.loc[persons_ci["metric"].eq("roc_auc")].iloc[0]
    st.caption(
        f"La MLP supera a la regla n_personas en ranking: ΔPR-AUC {pr_delta['delta']:+.4f} "
        f"[IC 95 % {pr_delta['ci_2_5']:+.4f}, {pr_delta['ci_97_5']:+.4f}] y ΔROC-AUC {roc_delta['delta']:+.4f} "
        f"[{roc_delta['ci_2_5']:+.4f}, {roc_delta['ci_97_5']:+.4f}]. Esta comparación de referencia es post-hoc; no reajusta la red."
    )

    st.subheader("Estabilidad temporal descriptiva")
    annual = design["annual"].copy()  # type: ignore[union-attr]
    annual_display = annual[["year", "n", "positives", "prevalence", "pr_auc", "roc_auc", "f1_multifatal", "precision_multifatal", "recall_multifatal"]].copy()
    annual_display.columns = ["Año", "n", "Positivos", "Prevalencia", "PR-AUC", "ROC-AUC", "F1", "Precisión", "Recall"]
    st.dataframe(annual_display.style.format({"Prevalencia": "{:.3f}", "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "F1": "{:.3f}", "Precisión": "{:.3f}", "Recall": "{:.3f}"}), width="stretch", hide_index=True)
    st.caption("2025 es parcial. La separación anual describe estabilidad; no se usó para elegir arquitectura, regularización, calibración ni umbral.")

    st.subheader("Calibración y clasificación")
    probabilities = evidence["probabilities"].copy()
    probabilities["bin"] = pd.cut(probabilities["calibrated_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    reliability_rows = []
    for interval, subset in probabilities.groupby("bin", observed=False):
        if subset.empty:
            continue
        n = len(subset)
        positives = int(subset["actual_multifatal"].sum())
        lower, upper = wilson_interval(positives, n)
        reliability_rows.append({"bin": str(interval), "predicha": subset["calibrated_probability"].mean(), "observada": positives / n, "n": n, "ci_inf": lower, "ci_sup": upper})
    reliability = pd.DataFrame(reliability_rows)
    reliability_fig = go.Figure()
    reliability_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=UNSUPPORTED, dash="dash"), name="Calibración ideal", hoverinfo="skip"))
    reliability_fig.add_trace(go.Scatter(
        x=reliability["predicha"], y=reliability["observada"], mode="markers",
        marker=dict(color=BLUE, size=np.clip(np.sqrt(reliability["n"]) * 1.5, 8, 24)), name="Bins observados",
        error_y=dict(type="data", array=reliability["ci_sup"] - reliability["observada"], arrayminus=reliability["observada"] - reliability["ci_inf"], color=MUTED),
        customdata=reliability[["n", "bin"]],
        hovertemplate="Predicha: %{x:.3f}<br>Observada: %{y:.3f}<br>n=%{customdata[0]:,}<br>Bin: %{customdata[1]}<extra></extra>",
    ))
    reliability_fig.update_layout(title="Confiabilidad de la probabilidad calibrada", legend_title_text="Referencia")
    reliability_fig.update_xaxes(title="Probabilidad media predicha", range=[0, 1])
    reliability_fig.update_yaxes(title="Frecuencia observada", range=[0, 1])

    confusion = evidence["confusion"]
    confusion = confusion[confusion["probability_scale"] == "calibrated"].pivot(index="actual", columns="predicted", values="count").reindex(index=[0, 1], columns=[0, 1])
    confusion_fig = go.Figure(go.Heatmap(
        z=confusion.to_numpy(), x=["Estimado: 1 fallecido", "Estimado: 2+ fallecidos"], y=["Real: 1 fallecido", "Real: 2+ fallecidos"],
        colorscale=[[0, "#EEF2F1"], [1, BLUE]], text=confusion.to_numpy(), texttemplate="%{text:,}",
        hovertemplate="Real: %{y}<br>Decisión: %{x}<br>n=%{z:,}<extra></extra>", showscale=False,
    ))
    confusion_fig.update_layout(title="Matriz de confusión · umbral calibrado")
    st.plotly_chart(_plot_layout(reliability_fig, height=440, legend=True), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(_plot_layout(confusion_fig, height=400), width="stretch", config={"displayModeBar": False})
    _table_fallback("confiabilidad", reliability.round(4), key="confiabilidad")
    confusion_table = (
        evidence["confusion"]
        .loc[lambda frame: frame["probability_scale"].eq("calibrated")]
        .copy()
        .assign(
            Clase_real=lambda frame: frame["actual"].map({0: "1 fallecido", 1: "2+ fallecidos"}),
            Decision=lambda frame: frame["predicted"].map({0: "1 fallecido", 1: "2+ fallecidos"}),
        )[["Clase_real", "Decision", "count"]]
        .rename(columns={"count": "Registros"})
    )
    _table_fallback("matriz de confusión", confusion_table, key="matriz_confusion")

    st.subheader("Curvas de ranking")
    y_true = probabilities["actual_multifatal"].to_numpy()
    y_score = probabilities["calibrated_probability"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    pr_fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color=ORANGE, width=2.5), name="MLP"))
    pr_fig.add_hline(y=float(y_true.mean()), line_dash="dash", line_color=MUTED, annotation_text="Prevalencia")
    pr_fig.update_layout(title="Curva Precision–Recall")
    pr_fig.update_xaxes(title="Recall", range=[0, 1])
    pr_fig.update_yaxes(title="Precisión", range=[0, 1])
    roc_fig = go.Figure(go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", line=dict(color=BLUE, width=2.5), name="MLP"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=UNSUPPORTED, dash="dash"), name="Azar", hoverinfo="skip"))
    roc_fig.update_layout(title="Curva ROC")
    roc_fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
    roc_fig.update_yaxes(title="Tasa de verdaderos positivos", range=[0, 1])
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_plot_layout(pr_fig, height=430), width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(_plot_layout(roc_fig, height=430, legend=True), width="stretch", config={"displayModeBar": False})
    ranking_tables = pd.concat(
        [
            pd.DataFrame({"Curva": "Precision–Recall", "Eje X": recall, "Eje Y": precision}),
            pd.DataFrame({"Curva": "ROC", "Eje X": false_positive_rate, "Eje Y": true_positive_rate}),
        ],
        ignore_index=True,
    )
    _table_fallback("puntos de curvas PR y ROC", ranking_tables.round(6), key="curvas_pr_roc")

    st.subheader("Explicabilidad")
    explainability = load_explainability_artifacts()
    groups = explainability["groups"].copy().sort_values("mean_abs_grouped_shap", ascending=True)
    provenance = explainability["provenance"]
    groups["Dirección media"] = groups["mean_signed_grouped_shap"].map(
        lambda value: "Aumenta el score bruto" if value > 0 else "Reduce el score bruto" if value < 0 else "Neutra"
    )
    explanation_fig = go.Figure()
    for direction, color in (("Aumenta el score bruto", ORANGE), ("Reduce el score bruto", BLUE), ("Neutra", UNSUPPORTED)):
        subset = groups[groups["Dirección media"] == direction]
        if subset.empty:
            continue
        explanation_fig.add_trace(go.Bar(
            x=subset["mean_abs_grouped_shap"],
            y=subset["raw_variable_group"],
            orientation="h",
            name=direction,
            marker_color=color,
            customdata=subset[["mean_signed_grouped_shap", "positive_contribution_share", "processed_feature_count"]],
            hovertemplate=(
                "<b>%{y}</b><br>Importancia global: %{x:.5f}<br>Contribución media firmada: %{customdata[0]:+.5f}"
                "<br>Proporción de contribuciones positivas: %{customdata[1]:.1%}"
                "<br>Features procesadas agrupadas: %{customdata[2]:.0f}<extra></extra>"
            ),
        ))
    explanation_fig.update_layout(
        title="Qué grupos de variables mueven el score de la MLP",
        barmode="overlay",
        legend_title_text="Dirección media en validación",
    )
    explanation_fig.update_xaxes(title="Media del valor SHAP absoluto agrupado")
    explanation_fig.update_yaxes(title=None)
    st.plotly_chart(_plot_layout(explanation_fig, height=560, legend=True), width="stretch", config={"displayModeBar": False})
    st.caption(
        f'{provenance["method"]} sobre el score sigmoide bruto · fondo: {provenance["background_sample_size"]} registros '
        f'2021–2022 · explicados: {provenance["explanation_sample_size"]} registros de validación 2023 · '
        f'semilla {provenance["seed"]}. Las one-hot y variables derivadas se sumaron dentro de su grupo interpretable.'
    )
    st.info(
        "Las contribuciones resumen asociaciones globales aprendidas por la MLP. El signo expresa el promedio de validación "
        "respecto del fondo de entrenamiento; la explicabilidad presentada corresponde al nivel global validado."
    )
    display_groups = groups.sort_values("mean_abs_grouped_shap", ascending=False).rename(columns={
        "raw_variable_group": "Grupo interpretable",
        "processed_feature_count": "Features agrupadas",
        "mean_abs_grouped_shap": "Importancia global",
        "mean_signed_grouped_shap": "Contribución media firmada",
        "positive_contribution_share": "Proporción positiva",
        "importance_share": "Participación de importancia",
    })
    _table_fallback(
        "importancia global agrupada",
        display_groups[["Grupo interpretable", "Features agrupadas", "Importancia global", "Contribución media firmada", "Proporción positiva", "Dirección media"]].round(5),
        key="explicabilidad_global",
    )
    with st.expander("Arquitectura y trazabilidad"):
        architecture = manifest["architecture"]
        st.json({
            "version": manifest["model_version"],
            "hidden_units": architecture["hidden_units"],
            "dropout": architecture["dropout"],
            "l2": architecture["l2"],
            "learning_rate": architecture["learning_rate"],
            "seed": architecture["seed"],
            "feature_count": manifest["feature_count"],
            "calibration": manifest["calibration"]["method"],
            "weights_frozen": architecture["weights_frozen"],
        })


def main() -> None:
    try:
        manifest = load_manifest()
        app_header(manifest)
        requested_slug = st.query_params.get("section", "panorama")
        label_by_slug = {slug: label for label, slug in SECTION_SLUGS.items()}
        initial_label = label_by_slug.get(str(requested_slug), "Panorama")
        selected = st.radio(
            "Sección",
            list(SECTION_SLUGS),
            index=list(SECTION_SLUGS).index(initial_label),
            horizontal=True,
            label_visibility="collapsed",
            key="section_navigation",
        )
        selected_slug = SECTION_SLUGS[selected]
        if str(requested_slug) != selected_slug:
            st.query_params["section"] = selected_slug
        pages = {
            "Panorama": lambda: overview_page(manifest),
            "Probar la red": estimate_page,
            "Explorar datos": explore_page,
            "Patrones regionales": regional_page,
            "Evidencia del modelo": lambda: evidence_page(manifest),
        }
        pages[selected]()
    except RuntimeArtifactError as exc:
        st.error(str(exc), icon="⚠️")
        st.info("La interfaz opera en modo estrictamente read-only. Corregí los artefactos fuera de la app y volvé a iniciar.")
        st.stop()


if __name__ == "__main__":
    main()
