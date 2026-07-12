from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "accident_nn_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    normalize_caracteristica,
    normalize_clase,
    normalize_clima,
    normalize_superficie,
    normalize_zona,
)

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "report" / "figures"
TABLES_DIR = ROOT / "report" / "tables"
SECTIONS_DIR = ROOT / "report" / "sections"


def multifatal_rate(series: pd.Series) -> float:
    return float(series.mean() * 100)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def prepare_eda_frame() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "base_limpia.parquet").copy()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["mes_periodo"] = df["FECHA"].dt.to_period("M").astype(str)
    df["mes"] = df["FECHA"].dt.month
    df["dia_semana"] = df["FECHA"].dt.dayofweek
    df["hora_entera"] = pd.to_numeric(df["hora_entera"], errors="coerce")
    df["target_label"] = df["target_multifatal"].map({0: "1 fallecido", 1: "2+ fallecidos"})
    df["clase_norm"] = df["CLASE"].apply(normalize_clase)
    df["zona_norm"] = df["ZONA"].apply(normalize_zona)
    df["clima_norm"] = df["CLIMA"].apply(normalize_clima)
    df["caracteristica_norm"] = df["CARACTERISTICA_VIA"].apply(normalize_caracteristica)
    df["superficie_norm"] = df["SUPERFICIE"].apply(normalize_superficie)
    return df


def generate_figures(df: pd.DataFrame) -> dict[str, object]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # fig01
    counts = df["target_label"].value_counts().reindex(["1 fallecido", "2+ fallecidos"])
    percentages = counts / counts.sum() * 100
    ax = counts.plot(kind="bar", color=["#3B82F6", "#EF4444"], figsize=(7, 4))
    ax.set_title("Distribución del target de letalidad")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Siniestros fatales")
    for i, (value, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, value, f"{value}\n{pct:.1f}%", ha="center", va="bottom")
    savefig("fig01_target_distribution.png")

    # fig02
    monthly = df.groupby("mes_periodo").size()
    ax = monthly.plot(marker="o", figsize=(10, 4), color="#2563EB")
    ax.set_title("Serie mensual de siniestros fatales")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Siniestros")
    ax.tick_params(axis="x", rotation=45)
    tick_positions = range(0, len(monthly), 3)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([monthly.index[i] for i in tick_positions])
    savefig("fig02_monthly_accidents.png")

    # fig03
    hourly_counts = df.dropna(subset=["hora_entera"]).groupby("hora_entera").size().reindex(range(24), fill_value=0)
    ax = hourly_counts.plot(kind="bar", figsize=(10, 4), color="#0F766E")
    ax.set_title("Siniestros fatales por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Siniestros")
    savefig("fig03_accidents_by_hour.png")

    # fig04
    hourly_rate = df.dropna(subset=["hora_entera"]).groupby("hora_entera")["target_multifatal"].mean().reindex(range(24)) * 100
    ax = hourly_rate.plot(marker="o", figsize=(10, 4), color="#DC2626")
    ax.set_title("Tasa de siniestros multifatales por hora")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Multifatales (%)")
    savefig("fig04_mortality_rate_by_hour.png")

    # fig05
    weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    weekday = (
        df.groupby("dia_semana")
        .agg(siniestros=("target_multifatal", "size"), multifatal=("target_multifatal", lambda s: s.mean() * 100))
        .reindex(range(7))
    )
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(weekday_names, weekday["siniestros"], color="#60A5FA", label="Siniestros")
    ax1.set_ylabel("Siniestros")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(weekday_names, weekday["multifatal"], color="#B91C1C", marker="o", label="Multifatales")
    ax2.set_ylabel("Multifatales (%)")
    ax1.set_title("Siniestros fatales y tasa multifatal por día de semana")
    savefig("fig05_weekday_accidents_mortality.png")

    # fig06
    clase_counts = df["clase_norm"].value_counts()
    ax = clase_counts.plot(kind="bar", figsize=(9, 4), color="#7C3AED")
    ax.set_title("Siniestros fatales por clase")
    ax.set_xlabel("Clase de siniestro")
    ax.set_ylabel("Siniestros")
    ax.tick_params(axis="x", rotation=35)
    savefig("fig06_accidents_by_modality.png")

    # fig07
    clase_rate = df.groupby("clase_norm")["target_multifatal"].mean().mul(100).sort_values(ascending=False)
    ax = clase_rate.plot(kind="bar", figsize=(9, 4), color="#EA580C")
    ax.set_title("Tasa multifatal por clase de siniestro")
    ax.set_xlabel("Clase de siniestro")
    ax.set_ylabel("Multifatales (%)")
    ax.tick_params(axis="x", rotation=35)
    savefig("fig07_mortality_by_modality.png")

    # fig08
    top_dept_counts = df["DEPARTAMENTO"].value_counts().head(15)
    ax = top_dept_counts.sort_values().plot(kind="barh", figsize=(8, 6), color="#0284C7")
    ax.set_title("Top-15 departamentos por siniestros fatales")
    ax.set_xlabel("Siniestros")
    savefig("fig08_top_departments_accidents.png")

    # fig09
    dept_summary = df.groupby("DEPARTAMENTO").agg(siniestros=("target_multifatal", "size"), multifatal=("target_multifatal", "mean"))
    dept_rate = dept_summary[dept_summary["siniestros"] >= 30]["multifatal"].mul(100).sort_values(ascending=False).head(15)
    ax = dept_rate.sort_values().plot(kind="barh", figsize=(8, 6), color="#BE123C")
    ax.set_title("Top-15 departamentos por tasa multifatal")
    ax.set_xlabel("Multifatales (%)")
    savefig("fig09_top_departments_mortality.png")

    # fig10
    top_roads = df["CODIGO_VIA"].value_counts().head(15)
    ax = top_roads.sort_values().plot(kind="barh", figsize=(8, 6), color="#4D7C0F")
    ax.set_title("Top-15 códigos de carretera por siniestros fatales")
    ax.set_xlabel("Siniestros")
    savefig("fig10_top_roads_accidents.png")

    # fig11: multifatal rate by pre-impact road/weather conditions
    condition_specs = [
        ("zona_norm", "Zona"),
        ("clima_norm", "Clima"),
        ("caracteristica_norm", "Característica de vía"),
        ("superficie_norm", "Superficie"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, title) in zip(axes.flat, condition_specs):
        rate = df.groupby(column)["target_multifatal"].mean().mul(100).sort_values(ascending=False)
        rate.plot(kind="bar", ax=ax, color="#B45309")
        ax.set_title(f"Tasa multifatal por {title.lower()}")
        ax.set_xlabel("")
        ax.set_ylabel("Multifatales (%)")
        ax.tick_params(axis="x", rotation=30)
    savefig("fig11_condition_rates.png")

    # fig12
    heatmap = df.dropna(subset=["hora_entera"]).pivot_table(index="mes", columns="hora_entera", values="target_multifatal", aggfunc="size", fill_value=0)
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap, cmap="YlOrRd")
    plt.title("Mapa de calor mes × hora: cantidad de siniestros fatales")
    plt.xlabel("Hora")
    plt.ylabel("Mes")
    savefig("fig12_month_hour_heatmap.png")

    # fig13
    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    ax = missing.plot(kind="bar", figsize=(10, 4), color="#64748B")
    ax.set_title("Porcentaje de faltantes por columna")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Faltantes (%)")
    ax.tick_params(axis="x", rotation=45)
    savefig("fig13_missing_values.png")

    # fig23: geographic scatter of fatal crashes
    geo = df.dropna(subset=["LATITUD", "LONGITUD"])
    fig, ax = plt.subplots(figsize=(7, 9))
    for label, color, size in [("1 fallecido", "#94A3B8", 4), ("2+ fallecidos", "#DC2626", 9)]:
        subset = geo[geo["target_label"] == label]
        ax.scatter(subset["LONGITUD"], subset["LATITUD"], s=size, c=color, alpha=0.5, label=label)
    ax.set_title("Ubicación de siniestros fatales 2021-2025")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend()
    ax.set_aspect("equal")
    savefig("fig23_geo_scatter.png")

    descriptive = df.describe(include="all").transpose()
    descriptive.to_csv(TABLES_DIR / "tab01_descriptive_statistics.csv")

    zona_rate = df.groupby("zona_norm")["target_multifatal"].mean().mul(100)
    clima_rate = df.groupby("clima_norm")["target_multifatal"].mean().mul(100)

    metrics = {
        "rows": int(df.shape[0]),
        "multifatal_count": int(df["target_multifatal"].sum()),
        "multifatal_rate": multifatal_rate(df["target_multifatal"]),
        "peak_month": str(monthly.idxmax()),
        "peak_month_count": int(monthly.max()),
        "peak_hour": int(hourly_counts.idxmax()),
        "peak_hour_count": int(hourly_counts.max()),
        "highest_hour_rate": int(hourly_rate.idxmax()),
        "highest_hour_rate_value": float(hourly_rate.max()),
        "top_clase": str(clase_counts.idxmax()),
        "top_clase_count": int(clase_counts.max()),
        "highest_clase_rate": str(clase_rate.idxmax()),
        "highest_clase_rate_value": float(clase_rate.max()),
        "top_department_accidents": str(top_dept_counts.idxmax()),
        "top_department_accidents_count": int(top_dept_counts.max()),
        "top_department_multifatal": str(dept_rate.idxmax()),
        "top_department_multifatal_rate": float(dept_rate.max()),
        "top_road": str(top_roads.idxmax()),
        "top_road_count": int(top_roads.max()),
        "rural_rate": float(zona_rate.get("RURAL", 0.0)),
        "urban_rate": float(zona_rate.get("URBANA", 0.0)),
        "rain_rate": float(clima_rate.get("LLUVIOSO", 0.0)),
        "clear_rate": float(clima_rate.get("DESPEJADO", 0.0)),
        "missing_top_column": str(missing.idxmax()),
        "missing_top_value": float(missing.max()),
    }
    (TABLES_DIR / "tab01_eda_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_interpretations(metrics: dict[str, object]) -> None:
    lines = [
        "# Hallazgos del Bloque C — EDA",
        "",
        f"H1: La clase multifatal (2+ fallecidos) representa {metrics['multifatal_rate']:.1f}% de los siniestros fatales (fig01). Esto confirma el desbalance y justifica evaluar con F1, recall y PR-AUC, no con accuracy aislada.",
        f"H2: El mes con más siniestros fatales fue {metrics['peak_month']} con {metrics['peak_month_count']} registros (fig02). La serie mensual muestra que la frecuencia no es constante, por lo que las variables temporales aportan señal al modelo.",
        f"H3: La hora con más siniestros fue las {metrics['peak_hour']:02d}:00 con {metrics['peak_hour_count']} registros (fig03). En cambio, la mayor tasa multifatal por hora aparece a las {metrics['highest_hour_rate']:02d}:00 con {metrics['highest_hour_rate_value']:.1f}% (fig04), mostrando que volumen y letalidad no son lo mismo.",
        "H4: La comparación por día de semana (fig05) muestra que la cantidad de siniestros y la tasa multifatal deben interpretarse juntas. Este patrón respalda incluir día de semana y fin de semana como características separadas.",
        f"H5: La clase de siniestro más frecuente fue {metrics['top_clase']} con {metrics['top_clase_count']} casos (fig06), mientras que la mayor tasa multifatal fue {metrics['highest_clase_rate']} con {metrics['highest_clase_rate_value']:.1f}% (fig07). Esto anticipa que la clase de siniestro debe ser una variable dominante.",
        f"H6: {metrics['top_department_accidents']} concentra la mayor cantidad de siniestros fatales ({metrics['top_department_accidents_count']}) (fig08), pero {metrics['top_department_multifatal']} lidera la tasa multifatal con {metrics['top_department_multifatal_rate']:.1f}% entre departamentos con al menos 30 casos (fig09). El ranking por volumen no equivale al ranking por letalidad.",
        f"H7: El código de carretera con mayor número de siniestros fue {metrics['top_road']} con {metrics['top_road_count']} registros (fig10). Esto respalda el uso de frecuencia de vía como feature, calculada solo en train para evitar fuga.",
        f"H8: Las condiciones pre-impacto separan el riesgo (fig11): zona rural {metrics['rural_rate']:.1f}% vs urbana {metrics['urban_rate']:.1f}%, y clima lluvioso {metrics['rain_rate']:.1f}% vs despejado {metrics['clear_rate']:.1f}%. Estas variables aportan contexto relevante para el modelo.",
        "H9: El mapa mes × hora (fig12) evidencia patrones combinados de temporalidad. Por eso no basta con incluir fecha u hora cruda: se derivan mes, franja y codificación cíclica.",
        f"H10: La columna con mayor porcentaje de faltantes en la base limpia fue {metrics['missing_top_column']} con {metrics['missing_top_value']:.1f}% (fig13). Las señales viales se excluyen del modelo por ese motivo; el resto admite imputación simple con flags.",
        "H11: El scatter geográfico (fig23) muestra los siniestros siguiendo los corredores viales del país, con eventos multifatales distribuidos en todas las regiones. Las coordenadas aportan señal espacial continua que complementa al departamento.",
        "",
        "## Interpretación por figura",
        "",
    ]
    figure_notes = {
        "fig01": "La distribución del target confirma que los siniestros con un solo fallecido dominan. Por eso la accuracy puede ser engañosa y se priorizan métricas de la clase multifatal.",
        "fig02": "La serie mensual permite observar variaciones temporales en la ocurrencia de siniestros fatales. Esto fundamenta las variables derivadas de fecha.",
        "fig03": "La distribución horaria identifica momentos de mayor volumen de siniestros. Volumen alto no implica automáticamente mayor letalidad.",
        "fig04": "La tasa multifatal por hora separa riesgo relativo de cantidad de eventos. Esta diferencia justifica nocturno, franja y codificación cíclica.",
        "fig05": "El día de semana combina volumen y letalidad en una misma lectura. La comparación sostiene usar día_semana y fin_de_semana como señales distintas.",
        "fig06": "La clase de siniestro muestra la composición de la mecánica del evento. Esta variable tiene relevancia directa para la letalidad.",
        "fig07": "La tasa multifatal por clase muestra que algunas mecánicas tienen riesgo relativo mayor aunque no sean las más frecuentes. Es uno de los hallazgos fuertes del EDA.",
        "fig08": "El ranking por departamento muestra concentración territorial del volumen. Sirve para contextualizar la siniestralidad por carga de eventos.",
        "fig09": "El ranking por tasa multifatal cambia la lectura territorial. Permite distinguir dónde ocurren más siniestros de dónde son relativamente más letales.",
        "fig10": "Los códigos de carretera más frecuentes sugieren concentración en rutas específicas. La frecuencia de vía se incorpora como señal sin tratar el código como número ordinal.",
        "fig11": "Las tasas por zona, clima, característica y superficie son las variables pre-impacto nuevas de esta fuente. Separan el riesgo estructural del contexto del siniestro.",
        "fig12": "El mapa mes × hora muestra patrones temporales cruzados. Refuerza que las features temporales deben capturar ciclos y no solo valores crudos.",
        "fig13": "La matriz de faltantes transparenta la calidad de datos usada. Justifica excluir señales viales y usar imputación con flags en el resto.",
        "fig23": "El scatter geográfico visualiza la red vial implícita en los datos. Las coordenadas entran al modelo como variables continuas estandarizadas.",
    }
    for fig, note in figure_notes.items():
        lines.append(f"- {fig}: {note}")

    (SECTIONS_DIR / "eda_hallazgos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_block_c() -> dict[str, object]:
    df = prepare_eda_frame()
    metrics = generate_figures(df)
    write_interpretations(metrics)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_block_c(), ensure_ascii=False, indent=2))
