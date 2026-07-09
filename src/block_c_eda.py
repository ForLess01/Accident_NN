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

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "report" / "figures"
TABLES_DIR = ROOT / "report" / "tables"
SECTIONS_DIR = ROOT / "report" / "sections"


def mortality_rate(series: pd.Series) -> float:
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
    df["target_label"] = df["target_mortal"].map({0: "No mortal", 1: "Mortal"})
    return df


def generate_figures(df: pd.DataFrame) -> dict[str, object]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # fig01
    counts = df["target_label"].value_counts().reindex(["No mortal", "Mortal"])
    percentages = counts / counts.sum() * 100
    ax = counts.plot(kind="bar", color=["#3B82F6", "#EF4444"], figsize=(7, 4))
    ax.set_title("Distribución del target de severidad")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Accidentes")
    for i, (value, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, value, f"{value}\n{pct:.1f}%", ha="center", va="bottom")
    savefig("fig01_target_distribution.png")

    # fig02
    monthly = df.groupby("mes_periodo").size()
    ax = monthly.plot(marker="o", figsize=(10, 4), color="#2563EB")
    ax.set_title("Serie mensual de accidentes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Accidentes")
    ax.tick_params(axis="x", rotation=45)
    savefig("fig02_monthly_accidents.png")

    # fig03
    hourly_counts = df.dropna(subset=["hora_entera"]).groupby("hora_entera").size().reindex(range(24), fill_value=0)
    ax = hourly_counts.plot(kind="bar", figsize=(10, 4), color="#0F766E")
    ax.set_title("Accidentes por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Accidentes")
    savefig("fig03_accidents_by_hour.png")

    # fig04
    hourly_rate = df.dropna(subset=["hora_entera"]).groupby("hora_entera")["target_mortal"].mean().reindex(range(24)) * 100
    ax = hourly_rate.plot(marker="o", figsize=(10, 4), color="#DC2626")
    ax.set_title("Tasa de mortalidad por hora")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Mortalidad (%)")
    savefig("fig04_mortality_rate_by_hour.png")

    # fig05
    weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    weekday = (
        df.groupby("dia_semana")
        .agg(accidentes=("target_mortal", "size"), mortalidad=("target_mortal", lambda s: s.mean() * 100))
        .reindex(range(7))
    )
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(weekday_names, weekday["accidentes"], color="#60A5FA", label="Accidentes")
    ax1.set_ylabel("Accidentes")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(weekday_names, weekday["mortalidad"], color="#B91C1C", marker="o", label="Mortalidad")
    ax2.set_ylabel("Mortalidad (%)")
    ax1.set_title("Accidentes y tasa de mortalidad por día de semana")
    savefig("fig05_weekday_accidents_mortality.png")

    # fig06
    modality_counts = df["MODALIDAD"].value_counts()
    ax = modality_counts.plot(kind="bar", figsize=(9, 4), color="#7C3AED")
    ax.set_title("Accidentes por modalidad")
    ax.set_xlabel("Modalidad")
    ax.set_ylabel("Accidentes")
    ax.tick_params(axis="x", rotation=35)
    savefig("fig06_accidents_by_modality.png")

    # fig07
    modality_rate = df.groupby("MODALIDAD")["target_mortal"].mean().mul(100).sort_values(ascending=False)
    ax = modality_rate.plot(kind="bar", figsize=(9, 4), color="#EA580C")
    ax.set_title("Tasa de mortalidad por modalidad")
    ax.set_xlabel("Modalidad")
    ax.set_ylabel("Mortalidad (%)")
    ax.tick_params(axis="x", rotation=35)
    savefig("fig07_mortality_by_modality.png")

    # fig08
    top_dept_counts = df["DEPARTAMENTO"].value_counts().head(15)
    ax = top_dept_counts.sort_values().plot(kind="barh", figsize=(8, 6), color="#0284C7")
    ax.set_title("Top-15 departamentos por accidentes")
    ax.set_xlabel("Accidentes")
    savefig("fig08_top_departments_accidents.png")

    # fig09
    dept_summary = df.groupby("DEPARTAMENTO").agg(accidentes=("target_mortal", "size"), mortalidad=("target_mortal", "mean"))
    dept_rate = dept_summary[dept_summary["accidentes"] >= 30]["mortalidad"].mul(100).sort_values(ascending=False).head(15)
    ax = dept_rate.sort_values().plot(kind="barh", figsize=(8, 6), color="#BE123C")
    ax.set_title("Top-15 departamentos por tasa de mortalidad")
    ax.set_xlabel("Mortalidad (%)")
    savefig("fig09_top_departments_mortality.png")

    # fig10
    top_roads = df["CODIGO_VIA"].value_counts().head(15)
    ax = top_roads.sort_values().plot(kind="barh", figsize=(8, 6), color="#4D7C0F")
    ax.set_title("Top-15 códigos de vía por accidentes")
    ax.set_xlabel("Accidentes")
    savefig("fig10_top_roads_accidents.png")

    # fig11
    top3_roads = df["CODIGO_VIA"].value_counts().head(3).index.tolist()
    subset = df[df["CODIGO_VIA"].isin(top3_roads)].dropna(subset=["KILOMETRO"])
    plt.figure(figsize=(10, 5))
    sns.histplot(data=subset, x="KILOMETRO", hue="CODIGO_VIA", bins=35, element="step", stat="count")
    plt.title("Distribución de kilómetro en las 3 vías con más accidentes")
    plt.xlabel("Kilómetro")
    plt.ylabel("Accidentes")
    savefig("fig11_kilometer_distribution_top_roads.png")

    # fig12
    heatmap = df.dropna(subset=["hora_entera"]).pivot_table(index="mes", columns="hora_entera", values="target_mortal", aggfunc="size", fill_value=0)
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap, cmap="YlOrRd")
    plt.title("Mapa de calor mes × hora: cantidad de accidentes")
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

    descriptive = df.describe(include="all").transpose()
    descriptive.to_csv(TABLES_DIR / "tab01_descriptive_statistics.csv")

    metrics = {
        "rows": int(df.shape[0]),
        "mortal_count": int(df["target_mortal"].sum()),
        "mortal_rate": mortality_rate(df["target_mortal"]),
        "peak_month": str(monthly.idxmax()),
        "peak_month_count": int(monthly.max()),
        "peak_hour": int(hourly_counts.idxmax()),
        "peak_hour_count": int(hourly_counts.max()),
        "highest_hour_rate": int(hourly_rate.idxmax()),
        "highest_hour_rate_value": float(hourly_rate.max()),
        "top_modality": str(modality_counts.idxmax()),
        "top_modality_count": int(modality_counts.max()),
        "highest_modality_rate": str(modality_rate.idxmax()),
        "highest_modality_rate_value": float(modality_rate.max()),
        "top_department_accidents": str(top_dept_counts.idxmax()),
        "top_department_accidents_count": int(top_dept_counts.max()),
        "top_department_mortality": str(dept_rate.idxmax()),
        "top_department_mortality_rate": float(dept_rate.max()),
        "top_road": str(top_roads.idxmax()),
        "top_road_count": int(top_roads.max()),
        "missing_top_column": str(missing.idxmax()),
        "missing_top_value": float(missing.max()),
    }
    (TABLES_DIR / "tab01_eda_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_interpretations(metrics: dict[str, object]) -> None:
    lines = [
        "# Hallazgos del Bloque C — EDA",
        "",
        f"H1: La clase mortal representa {metrics['mortal_rate']:.1f}% del dataset limpio (fig01). Esto confirma el desbalance y justifica evaluar con F1, recall y PR-AUC, no con accuracy aislada.",
        f"H2: El mes con más accidentes fue {metrics['peak_month']} con {metrics['peak_month_count']} registros (fig02). La serie mensual muestra que la frecuencia no es constante, por lo que las variables temporales aportan señal al modelo.",
        f"H3: La hora con más accidentes fue las {metrics['peak_hour']:02d}:00 con {metrics['peak_hour_count']} registros (fig03). En cambio, la mayor tasa de mortalidad por hora aparece a las {metrics['highest_hour_rate']:02d}:00 con {metrics['highest_hour_rate_value']:.1f}% (fig04), mostrando que volumen y severidad no son lo mismo.",
        "H4: La comparación por día de semana (fig05) muestra que la cantidad de accidentes y la tasa de mortalidad deben interpretarse juntas. Este patrón respalda incluir día de semana y fin de semana como características separadas.",
        f"H5: La modalidad más frecuente fue {metrics['top_modality']} con {metrics['top_modality_count']} accidentes (fig06), mientras que la mayor tasa de mortalidad fue {metrics['highest_modality_rate']} con {metrics['highest_modality_rate_value']:.1f}% (fig07). Esto anticipa que MODALIDAD debe ser una variable dominante.",
        f"H6: {metrics['top_department_accidents']} concentra la mayor cantidad de accidentes ({metrics['top_department_accidents_count']}) (fig08), pero {metrics['top_department_mortality']} lidera la tasa de mortalidad con {metrics['top_department_mortality_rate']:.1f}% entre departamentos con al menos 30 casos (fig09). El ranking por volumen no equivale al ranking por severidad.",
        f"H7: El código de vía con mayor número de accidentes fue {metrics['top_road']} con {metrics['top_road_count']} registros (fig10). Esto respalda el uso de frecuencia de vía como feature, calculada solo en train para evitar fuga.",
        "H8: La distribución de kilómetros en las tres vías con más accidentes (fig11) muestra concentración por tramos, no una dispersión homogénea. Esto justifica conservar KILOMETRO como variable numérica con imputación controlada.",
        "H9: El mapa mes × hora (fig12) evidencia patrones combinados de temporalidad. Por eso no basta con incluir fecha u hora cruda: se derivan mes, franja y codificación cíclica.",
        f"H10: La columna con mayor porcentaje de faltantes en la base limpia fue {metrics['missing_top_column']} con {metrics['missing_top_value']:.1f}% (fig13). La baja magnitud de faltantes permite usar imputaciones simples con flags en lugar de descartar masivamente datos.",
        "",
        "## Interpretación por figura",
        "",
    ]
    figure_notes = {
        "fig01": "La distribución del target confirma que la clase no mortal domina el dataset. Por eso la accuracy puede ser engañosa y se priorizan métricas de la clase mortal.",
        "fig02": "La serie mensual permite observar variaciones temporales en la ocurrencia de accidentes. Esto fundamenta las variables derivadas de fecha.",
        "fig03": "La distribución horaria identifica momentos de mayor volumen de accidentes. Volumen alto no implica automáticamente mayor severidad.",
        "fig04": "La tasa de mortalidad por hora separa riesgo relativo de cantidad de eventos. Esta diferencia justifica nocturno, franja y codificación cíclica.",
        "fig05": "El día de semana combina volumen y severidad en una misma lectura. La comparación sostiene usar día_semana y fin_de_semana como señales distintas.",
        "fig06": "La modalidad muestra la composición del tipo de accidente. Esta variable describe la mecánica del siniestro y tiene relevancia directa para severidad.",
        "fig07": "La tasa de mortalidad por modalidad muestra que algunas modalidades tienen riesgo relativo mayor aunque no sean las más frecuentes. Es uno de los hallazgos fuertes del EDA.",
        "fig08": "El ranking por departamento muestra concentración territorial del volumen. Sirve para contextualizar la siniestralidad por carga de eventos.",
        "fig09": "El ranking por tasa de mortalidad cambia la lectura territorial. Permite distinguir dónde ocurren más accidentes de dónde son relativamente más letales.",
        "fig10": "Los códigos de vía más frecuentes sugieren concentración en rutas específicas. La frecuencia de vía se incorpora como señal sin tratar el código como número ordinal.",
        "fig11": "La distribución por kilómetro sugiere tramos con concentración de accidentes en vías principales. Esto justifica no descartar KILOMETRO.",
        "fig12": "El mapa mes × hora muestra patrones temporales cruzados. Refuerza que las features temporales deben capturar ciclos y no solo valores crudos.",
        "fig13": "La matriz de faltantes transparenta la calidad de datos usada. La magnitud observada permite imputación con flags sin pérdida masiva de información.",
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
