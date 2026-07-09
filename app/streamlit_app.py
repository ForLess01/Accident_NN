from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app_inference import historical_comparison, load_clean_dataset, load_demo_cases, load_threshold, predict_records


st.set_page_config(
    page_title="Letalidad de siniestros fatales — Perú",
    page_icon="🛣️",
    layout="wide",
)


THEME_CSS = """
<style>
:root {
  --road-ink: #111827;
  --road-muted: #6B7280;
  --road-panel: #FFF7ED;
  --road-accent: #EA580C;
  --road-danger: #B91C1C;
  --road-safe: #0F766E;
}

html, body, [class*="css"] {
  font-family: 'Avenir Next', 'Gill Sans', 'Trebuchet MS', sans-serif;
}

.main .block-container {
  padding-top: 2rem;
  max-width: 1180px;
}

.hero {
  border: 1px solid rgba(17, 24, 39, .12);
  background:
    radial-gradient(circle at 12% 15%, rgba(234, 88, 12, .18), transparent 28%),
    linear-gradient(135deg, #FFF7ED 0%, #FFFFFF 48%, #F8FAFC 100%);
  border-radius: 28px;
  padding: 28px 32px;
  margin-bottom: 20px;
  box-shadow: 0 24px 60px rgba(17, 24, 39, .08);
}

.hero-kicker {
  font-family: 'Menlo', 'Consolas', monospace;
  color: var(--road-accent);
  letter-spacing: .08em;
  font-size: .78rem;
  font-weight: 700;
  text-transform: uppercase;
}

.hero-title {
  color: var(--road-ink);
  font-size: 2.3rem;
  line-height: 1.05;
  font-weight: 800;
  margin: 8px 0;
}

.hero-copy {
  color: var(--road-muted);
  font-size: 1.05rem;
  max-width: 780px;
}

.metric-card {
  border: 1px solid rgba(17, 24, 39, .10);
  border-radius: 20px;
  padding: 18px 20px;
  background: #FFFFFF;
  box-shadow: 0 14px 36px rgba(15, 23, 42, .06);
}

.metric-label {
  color: var(--road-muted);
  font-family: 'Menlo', 'Consolas', monospace;
  font-size: .75rem;
  text-transform: uppercase;
}

.metric-value {
  color: var(--road-ink);
  font-size: 1.8rem;
  font-weight: 800;
}

.note {
  border-left: 4px solid var(--road-accent);
  background: #FFF7ED;
  padding: 14px 16px;
  border-radius: 14px;
  color: #7C2D12;
}
</style>
"""

DEPARTAMENTO_OPTIONS = [
    "LIMA", "AREQUIPA", "LA LIBERTAD", "PUNO", "CUSCO", "JUNIN", "PIURA",
    "ANCASH", "CAJAMARCA", "AYACUCHO", "HUANCAVELICA", "ICA", "LAMBAYEQUE",
    "SAN MARTIN", "HUANUCO", "APURIMAC", "AMAZONAS", "LORETO", "PASCO",
    "TACNA", "MOQUEGUA", "TUMBES", "UCAYALI", "MADRE DE DIOS", "CALLAO",
]
ZONA_OPTIONS = ["RURAL", "URBANA"]
RED_VIAL_OPTIONS = ["NACIONAL", "DEPARTAMENTAL", "PROVINCIAL", "URBANO"]
TIPO_VIA_OPTIONS = ["CARRETERA", "AVENIDA", "CALLE", "VIA EXPRESA", "OTRO"]
CLASE_OPTIONS = ["CHOQUE", "DESPISTE", "ATROPELLO", "VOLCADURA", "ESPECIAL"]
CLIMA_OPTIONS = ["DESPEJADO", "NUBLADO", "LLUVIOSO", "NIEBLA", "OTRO"]
CARACTERISTICA_OPTIONS = ["TRAMO RECTO", "CURVA", "INTERSECCIÓN", "PUENTE", "OTRO"]
PERFIL_OPTIONS = ["PLANA", "INCLINADA"]
SUPERFICIE_OPTIONS = ["ASFALTADA", "AFIRMADO", "TROCHA", "CONCRETO"]


def app_header() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">ONSV · MLP tabular · demo local</div>
          <div class="hero-title">Clasificador de alta letalidad en siniestros viales fatales del Perú</div>
          <div class="hero-copy">
            Interfaz académica para estimar la probabilidad de que un siniestro fatal deje dos o más
            fallecidos, usando el mismo contrato de preprocesamiento del pipeline. No duplica features:
            consume <code>preparar_entrada()</code>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_clean_dataset() -> pd.DataFrame:
    return load_clean_dataset()


@st.cache_data(show_spinner=False)
def cached_demo_cases() -> pd.DataFrame:
    return load_demo_cases()


@st.cache_data(show_spinner=False)
def cached_shap_top5() -> pd.DataFrame:
    path = ROOT / "report" / "tables" / "tab08_shap_top5.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["feature", "mean_abs_shap"])


def comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    clean = comparison.dropna(subset=["tasa_multifatal_pct"]).copy()
    colors = ["#B91C1C" if label == "Score calibrado del modelo" else "#0F766E" for label in clean["comparador"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean["comparador"],
            y=clean["tasa_multifatal_pct"],
            mode="lines+markers+text",
            line={"color": "#111827", "width": 2},
            marker={"size": 13, "color": colors, "line": {"color": "#FFFFFF", "width": 2}},
            text=[f"{value:.1f}%" for value in clean["tasa_multifatal_pct"]],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Tasa multifatal: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Modelo vs tasas históricas observadas",
        xaxis_title="Comparador",
        yaxis_title="Probabilidad / tasa multifatal (%)",
        height=430,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-18)
    return fig


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    color = "#B91C1C" if probability >= threshold else "#0F766E"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            delta={"reference": threshold * 100, "suffix": "% umbral"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#CCFBF1"},
                    {"range": [threshold * 100, 100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "thickness": 0.75, "value": threshold * 100},
            },
            title={"text": "Probabilidad de 2+ fallecidos"},
        )
    )
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _default_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def prediction_tab() -> None:
    st.subheader("Predicción de alta letalidad")
    threshold = load_threshold()
    demo_cases = cached_demo_cases()

    selected_case = st.selectbox(
        "Cargar caso de prueba",
        ["Entrada manual", *demo_cases["caso_id"].tolist()],
    )
    if selected_case == "Entrada manual":
        defaults = {
            "FECHA": pd.Timestamp("2023-08-15").date(),
            "HORA": "17:30",
            "DEPARTAMENTO": "LIMA",
            "ZONA": "RURAL",
            "RED_VIAL": "NACIONAL",
            "TIPO_VIA": "CARRETERA",
            "CODIGO_VIA": "PE-1N",
            "CLASE": "CHOQUE",
            "CLIMA": "DESPEJADO",
            "CARACTERISTICA_VIA": "TRAMO RECTO",
            "PERFIL_VIA": "PLANA",
            "SUPERFICIE": "ASFALTADA",
        }
    else:
        row = demo_cases.loc[demo_cases["caso_id"] == selected_case].iloc[0]
        defaults = {
            "FECHA": pd.to_datetime(row["FECHA"]).date(),
            "HORA": str(row["HORA"]),
            "DEPARTAMENTO": str(row["DEPARTAMENTO"]),
            "ZONA": str(row["ZONA"]),
            "RED_VIAL": str(row["RED_VIAL"]),
            "TIPO_VIA": str(row["TIPO_VIA"]),
            "CODIGO_VIA": str(row["CODIGO_VIA"]),
            "CLASE": str(row["CLASE"]),
            "CLIMA": str(row["CLIMA"]),
            "CARACTERISTICA_VIA": str(row["CARACTERISTICA_VIA"]),
            "PERFIL_VIA": str(row["PERFIL_VIA"]),
            "SUPERFICIE": str(row["SUPERFICIE"]),
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        fecha = st.date_input("Fecha", value=defaults["FECHA"])
        hora = st.text_input("Hora (HH:MM)", value=defaults["HORA"])
        departamento = st.selectbox("Departamento", DEPARTAMENTO_OPTIONS, index=_default_index(DEPARTAMENTO_OPTIONS, defaults["DEPARTAMENTO"]))
        codigo_via = st.text_input("Código de carretera (opcional)", value=defaults["CODIGO_VIA"])
    with col2:
        zona = st.selectbox("Zona", ZONA_OPTIONS, index=_default_index(ZONA_OPTIONS, defaults["ZONA"]))
        red_vial = st.selectbox("Red vial", RED_VIAL_OPTIONS, index=_default_index(RED_VIAL_OPTIONS, defaults["RED_VIAL"]))
        tipo_via = st.selectbox("Tipo de vía", TIPO_VIA_OPTIONS, index=_default_index(TIPO_VIA_OPTIONS, defaults["TIPO_VIA"]))
        clase = st.selectbox("Clase de siniestro", CLASE_OPTIONS, index=_default_index(CLASE_OPTIONS, defaults["CLASE"]))
    with col3:
        clima = st.selectbox("Clima", CLIMA_OPTIONS, index=_default_index(CLIMA_OPTIONS, defaults["CLIMA"]))
        caracteristica = st.selectbox("Característica de vía", CARACTERISTICA_OPTIONS, index=_default_index(CARACTERISTICA_OPTIONS, defaults["CARACTERISTICA_VIA"]))
        perfil = st.selectbox("Perfil longitudinal", PERFIL_OPTIONS, index=_default_index(PERFIL_OPTIONS, defaults["PERFIL_VIA"]))
        superficie = st.selectbox("Superficie de calzada", SUPERFICIE_OPTIONS, index=_default_index(SUPERFICIE_OPTIONS, defaults["SUPERFICIE"]))

    record = pd.DataFrame(
        [
            {
                "FECHA": str(fecha),
                "HORA": hora,
                "DEPARTAMENTO": departamento,
                "ZONA": zona,
                "RED_VIAL": red_vial,
                "TIPO_VIA": tipo_via,
                "CODIGO_VIA": codigo_via.strip().upper() or "DESCONOCIDO",
                "CLASE": clase,
                "CLIMA": clima,
                "CARACTERISTICA_VIA": caracteristica,
                "PERFIL_VIA": perfil,
                "SUPERFICIE": superficie,
                "LATITUD": None,
                "LONGITUD": None,
            }
        ]
    )
    prediction = predict_records(record).iloc[0]
    score = float(prediction["score_riesgo_multifatal"])
    raw_probability = float(prediction["probabilidad_multifatal"])
    label = str(prediction["clasificacion"])
    comparison = historical_comparison(record.iloc[0], score)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(probability_gauge(score, threshold), width="stretch")
    with right:
        st.markdown('<div class="metric-card"><div class="metric-label">Clasificación</div><div class="metric-value">{}</div></div>'.format(label), unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="metric-card"><div class="metric-label">Umbral (score crudo)</div><div class="metric-value">{:.2f}</div></div>'.format(threshold), unsafe_allow_html=True)
        st.write("")
        st.markdown(
            '<div class="note">El gauge muestra la probabilidad calibrada (isotónica sobre validación). '
            'La clasificación usa el score crudo {:.3f} contra el umbral. '
            'Es una estimación académica; no reemplaza evaluación vial profesional.</div>'.format(raw_probability),
            unsafe_allow_html=True,
        )

    st.markdown("#### Comparación con datos reales")
    st.plotly_chart(comparison_chart(comparison), width="stretch")
    st.caption(
        "La línea no es una comprobación individual del caso manual. Compara la probabilidad calibrada "
        "contra tasas multifatales observadas en grupos reales del dataset ONSV."
    )
    st.dataframe(
        comparison.assign(tasa_multifatal_pct=lambda data: data["tasa_multifatal_pct"].round(2)),
        width="stretch",
        hide_index=True,
    )


def eda_tab() -> None:
    st.subheader("Análisis exploratorio")
    st.markdown("Las figuras se generaron en el Bloque C y se usan aquí como tablero de lectura rápida.")
    figures = [
        ("fig01_target_distribution.png", "Distribución del target"),
        ("fig02_monthly_accidents.png", "Serie mensual"),
        ("fig07_mortality_by_modality.png", "Tasa multifatal por clase"),
        ("fig11_condition_rates.png", "Tasas por condiciones pre-impacto"),
        ("fig12_month_hour_heatmap.png", "Mapa mes × hora"),
        ("fig23_geo_scatter.png", "Ubicación de siniestros"),
    ]
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, (filename, caption) in zip(cols, figures[i : i + 2]):
            path = ROOT / "report" / "figures" / filename
            with col:
                if path.exists():
                    st.image(str(path), caption=caption, width="stretch")

    findings_path = ROOT / "report" / "sections" / "eda_hallazgos.md"
    if findings_path.exists():
        with st.expander("Ver H1–H11"):
            st.markdown(findings_path.read_text(encoding="utf-8"))


def risk_tab() -> None:
    st.subheader("Letalidad por departamento")
    df = cached_clean_dataset()
    dept = (
        df.groupby("DEPARTAMENTO")
        .agg(siniestros=("target_multifatal", "size"), multifatal=("target_multifatal", "mean"))
        .reset_index()
    )
    dept["multifatal"] = dept["multifatal"] * 100
    dept = dept.sort_values("multifatal", ascending=False)

    filtered = dept[dept["siniestros"] >= 30].copy()
    fig = px.bar(
        filtered.head(15).sort_values("multifatal"),
        x="multifatal",
        y="DEPARTAMENTO",
        orientation="h",
        color="siniestros",
        color_continuous_scale="OrRd",
        labels={"multifatal": "Tasa multifatal (%)", "DEPARTAMENTO": "Departamento", "siniestros": "Siniestros fatales"},
        title="Ranking departamental por tasa de siniestros multifatales",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(filtered, width="stretch", hide_index=True)

    geo_path = ROOT / "data" / "geo" / "peru_departamentos_simple.geojson"
    if geo_path.exists():
        try:
            with open(geo_path, encoding="utf-8") as geojson_file:
                geojson = json.load(geojson_file)
            available = {feature["properties"]["NOMBDEP"] for feature in geojson["features"]}
            map_df = dept[dept["DEPARTAMENTO"].isin(available)].copy()
            if not map_df.empty:
                st.markdown("#### Mapa coroplético del Perú")
                choropleth = px.choropleth(
                    map_df,
                    geojson=geojson,
                    locations="DEPARTAMENTO",
                    featureidkey="properties.NOMBDEP",
                    color="multifatal",
                    color_continuous_scale="OrRd",
                    labels={"multifatal": "Tasa multifatal (%)", "DEPARTAMENTO": "Departamento"},
                    hover_data={"siniestros": True, "multifatal": ":.1f"},
                    title="Tasa de siniestros multifatales por departamento",
                )
                choropleth.update_geos(
                    fitbounds="locations",
                    visible=False,
                    showcountries=False,
                    showsubunits=False,
                )
                choropleth.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(choropleth, width="stretch")
                col_top, col_bot = st.columns(2)
                with col_top:
                    st.caption("Color: tasa multifatal (%) por departamento.")
                with col_bot:
                    st.caption("Fuente del GeoJSON: repositorio público `juaneladio/peru-geojson`.")
            else:
                st.warning("No se encontraron departamentos compatibles con el GeoJSON.")
        except (OSError, ValueError) as exc:
            st.warning(f"No se pudo cargar el GeoJSON: {exc}. El ranking de barras sigue disponible.")
    else:
        st.info("GeoJSON no presente en data/geo/. El ranking de barras cumple la función obligatoria.")


def about_tab() -> None:
    st.subheader("Sobre el modelo")
    summary_path = ROOT / "report" / "tables" / "tab05_test_summary.json"
    threshold = load_threshold()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cols = st.columns(4)
        metrics = [
            ("F1 multifatal", summary["f1_multifatal"]),
            ("Recall multifatal", summary["recall_multifatal"]),
            ("PR-AUC", summary["pr_auc"]),
            ("ROC-AUC", summary["roc_auc"]),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value:.4f}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        - Problema: dado un siniestro fatal, estimar si deja 2+ fallecidos (alta letalidad).
        - Modelo: MLP tabular guardado en `models/letalidad_nn.keras`.
        - Umbral de decisión: `{threshold:.2f}`, seleccionado en validación.
        - Calibración: isotónica ajustada en validación (`models/calibrator.pkl`).
        - Preprocesamiento: `src.preprocessing.preparar_entrada()`.
        - Dataset: siniestros fatales 2021–2025 (preliminar), ONSV.
        - Integrantes: Rendo y Yimmy.
        """
    )
    shap_top = cached_shap_top5()
    if not shap_top.empty:
        st.markdown("#### Top-5 factores globales por SHAP")
        st.dataframe(shap_top, width="stretch", hide_index=True)


def main() -> None:
    app_header()
    tabs = st.tabs(["Predicción", "EDA", "Letalidad por departamento", "Sobre el modelo"])
    with tabs[0]:
        prediction_tab()
    with tabs[1]:
        eda_tab()
    with tabs[2]:
        risk_tab()
    with tabs[3]:
        about_tab()


if __name__ == "__main__":
    main()
