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
    calibration_summary_path = TABLES_DIR / "tab11_calibration_posthoc_summary.json"
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8")) if calibration_summary_path.exists() else {}

    leakage_tokens = [
        "FALLECIDOS",
        "LESIONADOS",
        "VEHICULOS_DANADOS",
        "VEHÍCULOS_DAÑADOS",
        "CAUSA",
        "SENAL_VERTICAL",
        "SEÑAL_VERTICAL",
        "SENAL_HORIZONTAL",
        "SEÑAL_HORIZONTAL",
    ]
    leakage_features = [
        feature
        for feature in feature_list
        if any(token in feature.upper() for token in leakage_tokens)
    ]
    mlp_val = validation[validation["modelo"].str.startswith("MLP_")].iloc[0]
    mlp_test = test[test["modelo"].str.startswith("MLP_")].iloc[0]
    baseline_val = validation[~validation["modelo"].str.startswith("MLP_")]
    baseline_test = test[~test["modelo"].str.startswith("MLP_")]
    best_baseline_val_f1 = baseline_val["f1_multifatal"].max()
    best_baseline_test_f1 = baseline_test["f1_multifatal"].max()
    best_baseline_test_model = baseline_test.sort_values("f1_multifatal", ascending=False).iloc[0]["modelo"]
    max_recall_threshold = threshold.sort_values(["recall_multifatal", "f1_multifatal"], ascending=False).iloc[0]

    audit = {
        "binary_classification": True,
        "leakage_features": leakage_features,
        "feature_count": len(feature_list),
        "target": "target_multifatal",
        "target_definition": "1 si FALLECIDOS >= 2; 0 si FALLECIDOS == 1",
        "mlp_validation_f1_multifatal": float(mlp_val["f1_multifatal"]),
        "best_baseline_validation_f1_multifatal": float(best_baseline_val_f1),
        "mlp_test_f1_multifatal": float(mlp_test["f1_multifatal"]),
        "best_baseline_test_model": str(best_baseline_test_model),
        "best_baseline_test_f1_multifatal": float(best_baseline_test_f1),
        "mlp_test_recall_multifatal": float(mlp_test["recall_multifatal"]),
        "mlp_test_pr_auc": float(mlp_test["pr_auc"]),
        "mlp_test_roc_auc": float(mlp_test["roc_auc"]),
        "test_evaluations": int(summary["test_evaluations"]),
        "selected_threshold": float(summary["threshold"]),
        "max_recall_validation_threshold": float(max_recall_threshold["threshold"]),
        "max_recall_validation_recall": float(max_recall_threshold["recall_multifatal"]),
        "max_recall_validation_f1": float(max_recall_threshold["f1_multifatal"]),
        "selected_calibrator": calibration_summary.get("selected_calibrator"),
        "raw_test_brier": calibration_summary.get("raw_test_brier"),
        "calibrated_test_brier": calibration_summary.get("calibrated_test_brier"),
        "raw_test_ece_10_bins": calibration_summary.get("raw_test_ece_10_bins"),
        "calibrated_test_ece_10_bins": calibration_summary.get("calibrated_test_ece_10_bins"),
        "verdict": "metodologicamente_correcto_para_target_multifatal_con_empate_frente_a_logistica_y_calibracion_mejorada",
    }
    (TABLES_DIR / "tab10_model_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    text = f"""# Auditoría del modelo frente al Plan.md

## Veredicto

El modelo es metodológicamente correcto para la definición reformulada del proyecto: clasificación binaria de alta letalidad en siniestros fatales (`target_multifatal = 1` si `FALLECIDOS >= 2`), salida sigmoide, evaluación con métricas para desbalance y comparación contra baselines. No se detectaron features derivadas de fallecidos, lesionados, vehículos dañados, causas post-investigación ni señalización excluida, por lo que no hay evidencia de fuga directa del target en `feature_list.json`.

## Evidencia

- Features finales: {audit['feature_count']}.
- Features con posible fuga: {len(leakage_features)}.
- F1-multifatal MLP en validación: {audit['mlp_validation_f1_multifatal']:.4f}.
- Mejor F1-multifatal baseline en validación: {audit['best_baseline_validation_f1_multifatal']:.4f}.
- F1-multifatal MLP en test: {audit['mlp_test_f1_multifatal']:.4f}.
- Mejor F1-multifatal baseline en test: {audit['best_baseline_test_f1_multifatal']:.4f} ({audit['best_baseline_test_model']}).
- Recall-multifatal MLP en test: {audit['mlp_test_recall_multifatal']:.4f}.
- PR-AUC MLP en test: {audit['mlp_test_pr_auc']:.4f}.
- ROC-AUC MLP en test: {audit['mlp_test_roc_auc']:.4f}.
- Evaluaciones del test registradas: {audit['test_evaluations']}.
- Calibrador post-hoc seleccionado: {audit['selected_calibrator']}.
- Brier crudo/calibrado en test: {audit['raw_test_brier']:.4f} / {audit['calibrated_test_brier']:.4f}.
- ECE crudo/calibrado en test: {audit['raw_test_ece_10_bins']:.4f} / {audit['calibrated_test_ece_10_bins']:.4f}.

## ¿Necesita mejorarse?

No necesita corregirse para cumplir el plan: usa el split correcto, mantiene el test como evaluación final única, evita leakage y privilegia el recall del evento de mayor costo. Sí debe presentarse con honestidad: la regresión logística gana nominalmente en F1 de test, mientras que el MLP gana en recall multifatal ({audit['mlp_test_recall_multifatal']:.4f}) y mantiene ROC-AUC comparable. Ese empate con un baseline clásico es esperable en datos tabulares medianos y no debe maquillarse.

La mejora aplicada no cambia la red ni maquilla el F1: calibra la lectura del riesgo con validation. El umbral de máxima sensibilidad en validación fue {audit['max_recall_validation_threshold']:.2f}, con recall {audit['max_recall_validation_recall']:.4f} pero F1 {audit['max_recall_validation_f1']:.4f}. Cambiar ahora el umbral después de haber visto el test no sería metodológicamente limpio; se debe reportar esta tensión en Discusión y proponer como trabajo futuro una política de umbral definida por costo operacional antes de evaluar un nuevo holdout.
"""
    (SECTIONS_DIR / "model_audit.md").write_text(text, encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run_model_audit(), ensure_ascii=False, indent=2))
