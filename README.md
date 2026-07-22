# Accident_NN - entrega académica definitiva

Red neuronal reproducible para clasificar retrospectivamente siniestros viales **ya fatales** del Perú como un fallecido o dos o más fallecidos. Autores: **Rendo Alfonte Tarqui** y **Yimmy Yeyson Cuyo Zamata**.

## Resultado verificable

- Fuente: ONSV 2021-2025 (2024-2025 preliminar), 9.104 registros.
- Entrada: 21 campos crudos, **169 características procesadas**.
- Red única: **169 -> Dense(64, ReLU) -> Dropout(0,35) -> Dense(32, ReLU) -> Dropout(0,35) -> Dense(1, sigmoide)**.
- Parámetros entrenables: **12.993**.
- Regularización/técnicas reales: L2 $3e-4$, dropout, pesos de clase, Adam, early stopping y ReduceLROnPlateau durante selección; calibración Platt OOF.
- Selección de arquitectura: fit 2021, selección 2022; reajuste final en 2021-2022 durante 14 épocas fijas; calibración y umbrales en 2023.
- Referencia histórica ya consultada 2024-2025: PR-AUC 0,3254; ROC-AUC 0,7916; F1 calibrado 0,3989; Brier 0,0777; ECE 0,0133.

El desempeño es útil como demostración académica, pero **no es excelente ni confirmatorio**: no existe cohorte externa o prospectiva intacta. La finalidad de esta entrega no convierte 2024-2025 en un test nuevo.

## Corrección científica central

La auditoría ejecutable demostró que en 9.104/9.104 siniestros el conteo `GRAVEDAD=FALLECIDO` de PERSONAS coincide con `FALLECIDOS`, y reconstruye `target_multifatal`. Por eso se excluyó **toda** la familia PERSONAS: `n_personas`, pasajeros, peatones, conductores fugados, edad media y su indicador. PERSONAS solo permanece como evidencia de linaje/proxy.

VEHICULOS se conserva exclusivamente bajo el alcance **clasificación histórica retrospectiva de registros consolidados**. Los libros públicos no incluyen timestamps por campo; no se afirma disponibilidad al notificar ni operación en tiempo real.

## Ejecutar

```bash
cd /Users/rendoaltar/Developer/Accident_NN
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Abrí la URL local indicada por Streamlit. La app no entrena ni modifica artefactos.

## Reproducir artefactos

```bash
python src/block_b_dataset_audit.py
python src/block_e_modeling.py
python src/final_model_bundle.py
python src/temporal_diagnostics.py
python src/validation_design_audit.py
python src/final_paired_comparison.py
python src/final_explainability.py
python src/demo_cases.py
python src/final_evaluation_figures.py
python scripts/execute_notebooks.py
python scripts/build_report.py
python scripts/check_release.py --local-content
```

La reconstrucción del modelo es deliberadamente explícita y costosa. La app ordinaria solo consume `models/final/` con verificación de hashes.

## Evidencia principal

- `data/raw/source_manifest.json`: URLs, tamaños, hojas y SHA-256 oficiales.
- `docs/data_provenance.md`: procedencia, disponibilidad y alcance.
- `report/tables/personas_target_proxy_identity.*`: prueba del proxy.
- `report/tables/temporal_nested_*`: dos diagnósticos rolling con fit/selección/calibración/outer disjuntos dentro de cada fold.
- `report/tables/model_learning_curves.csv` y `model_generalization_gap.json`: curvas y brecha con métricas no ponderadas comparables.
- `report/tables/final_explainability_stability.csv`: estabilidad SHAP en tres muestras/semillas; se interpretan bandas, no orden exacto.
- `data/processed/demo_cases.csv`: TN, FP, FN limítrofe y TP reales de 2023, más clon sintético con código de vía no visto.
- `report/build_manifest.json`: frescura del PDF por contenido, no por fechas del filesystem.

## Incertidumbre

Los IC bootstrap remuestrean filas de la referencia con pipeline y predicciones congelados. No incluyen incertidumbre de entrenamiento/selección, dependencia temporal o espacial, consultas repetidas, generalización externa/futura ni un intervalo de predicción individual. Los IC Wilson de la app describen tasas agregadas, no incertidumbre por caso.
