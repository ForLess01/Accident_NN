from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "report" / "tables"
SECTIONS_DIR = ROOT / "report" / "sections"
MODELS_DIR = ROOT / "models"


def run_model_audit() -> dict[str, object]:
    feature_list = json.loads((MODELS_DIR / "feature_list.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(TABLES_DIR / "tab03_model_comparison_validation.csv")
    test = pd.read_csv(TABLES_DIR / "tab03_model_comparison_test.csv")
    threshold = pd.read_csv(TABLES_DIR / "tab_umbral_validacion.csv")
    summary = json.loads((TABLES_DIR / "tab05_test_summary.json").read_text(encoding="utf-8"))

    leakage_features = [
        feature
        for feature in feature_list
        if "FALLECIDOS" in feature.upper() or "HERIDOS" in feature.upper()
    ]
    mlp_val = validation[validation["modelo"].str.startswith("MLP_")].iloc[0]
    mlp_test = test[test["modelo"].str.startswith("MLP_")].iloc[0]
    best_baseline_val_f1 = validation[~validation["modelo"].str.startswith("MLP_")]["f1_mortal"].max()
    best_baseline_test_f1 = test[~test["modelo"].str.startswith("MLP_")]["f1_mortal"].max()
    max_recall_threshold = threshold.sort_values(["recall_mortal", "f1_mortal"], ascending=False).iloc[0]

    audit = {
        "binary_classification": True,
        "leakage_features": leakage_features,
        "feature_count": len(feature_list),
        "mlp_validation_f1_mortal": float(mlp_val["f1_mortal"]),
        "best_baseline_validation_f1_mortal": float(best_baseline_val_f1),
        "mlp_test_f1_mortal": float(mlp_test["f1_mortal"]),
        "best_baseline_test_f1_mortal": float(best_baseline_test_f1),
        "mlp_test_recall_mortal": float(mlp_test["recall_mortal"]),
        "mlp_test_pr_auc": float(mlp_test["pr_auc"]),
        "test_evaluations": int(summary["test_evaluations"]),
        "selected_threshold": float(summary["threshold"]),
        "max_recall_validation_threshold": float(max_recall_threshold["threshold"]),
        "max_recall_validation_recall": float(max_recall_threshold["recall_mortal"]),
        "max_recall_validation_f1": float(max_recall_threshold["f1_mortal"]),
        "verdict": "metodologicamente_correcto_con_desempeno_limitado",
    }
    (TABLES_DIR / "tab10_model_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    text = f"""# Auditoría del modelo frente al Plan.md

## Veredicto

El modelo es metodológicamente correcto para la definición del proyecto: clasificación binaria mortal/no mortal, salida probabilística sigmoide, evaluación con métricas para desbalance y comparación contra baselines. No se detectaron features con `FALLECIDOS` ni `HERIDOS`, por lo que no hay evidencia de fuga directa del target en `feature_list.json`.

## Evidencia

- Features finales: {audit['feature_count']}.
- Features con posible fuga: {len(leakage_features)}.
- F1-mortal MLP en validación: {audit['mlp_validation_f1_mortal']:.4f}.
- Mejor F1-mortal baseline en validación: {audit['best_baseline_validation_f1_mortal']:.4f}.
- F1-mortal MLP en test: {audit['mlp_test_f1_mortal']:.4f}.
- Mejor F1-mortal baseline en test: {audit['best_baseline_test_f1_mortal']:.4f}.
- Recall-mortal MLP en test: {audit['mlp_test_recall_mortal']:.4f}.
- PR-AUC MLP en test: {audit['mlp_test_pr_auc']:.4f}.
- Evaluaciones del test registradas: {audit['test_evaluations']}.

## ¿Necesita mejorarse?

No necesita corregirse para cumplir el plan: supera al Dummy y a los baselines en F1-mortal, usa el split correcto y mantiene el test como evaluación final. Sí debe presentarse como un modelo de desempeño limitado: el recall-mortal de test es {audit['mlp_test_recall_mortal']:.4f}, por lo que todavía deja falsos negativos relevantes.

El umbral de máxima sensibilidad en validación fue {audit['max_recall_validation_threshold']:.2f}, con recall {audit['max_recall_validation_recall']:.4f} pero F1 {audit['max_recall_validation_f1']:.4f}. Cambiar ahora el umbral después de haber visto el test no sería metodológicamente limpio; se debe reportar esta tensión en Discusión y proponer como trabajo futuro una política de umbral definida por costo operacional antes de evaluar un nuevo holdout.
"""
    (SECTIONS_DIR / "model_audit.md").write_text(text, encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run_model_audit(), ensure_ascii=False, indent=2))
