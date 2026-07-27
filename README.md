# Clasificación de alta letalidad en siniestros viales fatales del Perú

Red neuronal reproducible para clasificar retrospectivamente siniestros viales **ya fatales** del Perú según registren un fallecido o dos o más fallecidos (multifatalidad), a partir de los registros abiertos del Observatorio Nacional de Seguridad Vial (ONSV) 2021--2025.

- **Autores:** Rendo Alfonte Tarqui; Yimmy Yeyson Cuyo Zamata
- **Curso:** Redes Neuronales --- Segunda Unidad
- **Repositorio:** <https://github.com/ForLess01/Accident_NN>

## Resumen técnico

- Fuente: ONSV 2021--2025 (2024--2025 preliminar), 9 104 registros verificados por SHA-256.
- Entrada: 21 campos crudos transformados en **169 características** procesadas.
- Red única: **169 → Dense(64, ReLU) → Dropout(0,35) → Dense(32, ReLU) → Dropout(0,35) → Dense(1, sigmoide)**, con **12 993 parámetros entrenables**.
- Técnicas aplicadas: BCE con pesos de clase, Adam, L2 $3\times10^{-4}$, dropout, early stopping y ReduceLROnPlateau durante la selección; calibración Platt con validación OOF.
- Protocolo temporal: la arquitectura se selecciona con ajuste 2021 y comparación 2022; el modelo se reajusta una única vez en 2021--2022 durante 14 épocas fijas; la calibración y los umbrales se validan en 2023.
- Referencia histórica 2024--2025 (ya consultada): PR-AUC 0,3254; ROC-AUC 0,7916; F1 calibrado 0,3989; Brier 0,0777; ECE 0,0133.

El desempeño es moderado y se reporta como demostración académica: no existe una cohorte externa o prospectiva intacta, por lo que la referencia 2024--2025 no constituye un test confirmatorio.

## Decisión metodológica central: exclusión de PERSONAS

Una auditoría ejecutable demostró que, en los 9 104 siniestros, el conteo `GRAVEDAD=FALLECIDO` de la tabla PERSONAS coincide con `FALLECIDOS` y reconstruye `target_multifatal`. Por esa razón se excluye **toda** la familia de agregados PERSONAS de los predictores (`n_personas`, pasajeros, peatones, conductores fugados, edad media y su indicador). PERSONAS permanece en el repositorio únicamente como evidencia de linaje del proxy (`report/tables/personas_target_proxy_identity.*`).

Los agregados VEHICULOS se conservan exclusivamente bajo el alcance de **clasificación histórica retrospectiva de registros consolidados**: los libros públicos no incluyen timestamps por campo, por lo que no se afirma disponibilidad al momento de la notificación ni operación en tiempo real.

## Estructura del repositorio

```text
Accident_NN/
├── app/                  # Interfaz gráfica Streamlit (solo lectura del bundle)
├── data/
│   ├── raw/              # Libros oficiales ONSV + diccionario + manifiesto SHA-256
│   ├── processed/        # base_limpia.parquet y casos de demostración
│   └── geo/              # GeoJSON de departamentos del Perú
├── docs/                 # Procedencia y disponibilidad de los datos
├── models/final/         # Bundle canónico: modelo, encoders, calibrador, umbrales
├── notebooks/            # modelo.ipynb: cuaderno académico integral (Colab)
├── report/               # Informe LaTeX, figuras, tablas y PDF final
├── scripts/              # Construcción del informe y gate de release
├── src/                  # Pipeline: auditoría, EDA, entrenamiento, evaluación
└── tests/                # Suite de regresión (pytest)
```

## Ejecución de la interfaz

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # en macOS: requirements-macos.txt
streamlit run app/streamlit_app.py
```

La aplicación abre en la URL local indicada por Streamlit. No entrena ni modifica artefactos: consume `models/final/` con verificación de hashes.

## Cuaderno académico

`notebooks/modelo.ipynb` documenta el proyecto completo en un único cuaderno ejecutable (compatible con Google Colab): verificación de fuentes, análisis exploratorio, partición cronológica, ingeniería de características, arquitectura, entrenamiento, calibración, evaluación y explicabilidad. La limpieza inicial de datos se realiza con los scripts de `src/` y queda documentada en el propio cuaderno.

## Reproducción de artefactos

El orden canónico de reproducción es el siguiente:

```bash
python src/source_provenance.py        # 1. Verifica fuentes y prueba el proxy PERSONAS
python src/block_b_dataset_audit.py    # 2. Genera base_limpia.parquet (sin PERSONAS)
python src/block_c_eda.py              # 3. Figuras y tablas del análisis exploratorio
python src/block_e_modeling.py         # 4. Selección 2021/2022 y reajuste 2021-2022
python src/final_model_bundle.py       # 5. Calibración y umbrales en 2023; referencia 2024-2025
python src/temporal_diagnostics.py     # 6. Folds rolling internos con roles disjuntos
python src/validation_design_audit.py  # 7. Auditoría de regularización y número de redes
python src/embedding_design_audit.py   # 7b. Auditoría de representación: embeddings vs one-hot
python src/vehicle_enrichment_audit.py # 7c. Auditoría de enriquecimiento del contrato VEHICULOS
python src/capacity_design_audit.py    # 7d. Auditoría de capacidad: ancho, profundidad, épocas y muestra
python src/final_paired_comparison.py  # 8. Bootstrap pareado contra líneas base
python src/final_explainability.py     # 9. Gradient SHAP y estabilidad
python src/demo_cases.py               # 10. Casos de demostración sellados
python src/final_evaluation_figures.py # 11. Figuras finales de evidencia
python scripts/execute_notebooks.py    # 12. Ejecuta el cuaderno académico
python scripts/build_report.py         # 13. Compila y sella el PDF por contenido
python scripts/check_release.py --local-content  # 14. Gate final de entrega
```

La reconstrucción del modelo es deliberadamente explícita y costosa. No existe un test externo o prospectivo intacto: un avance confirmatorio requiere una cohorte futura o externa no observada.

## Evidencia principal

- `data/raw/source_manifest.json`: URLs, tamaños, hojas y SHA-256 oficiales.
- `docs/data_provenance.md`: procedencia, disponibilidad y alcance de las fuentes.
- `report/tables/personas_target_proxy_identity.*`: prueba ejecutable del proxy.
- `report/tables/temporal_nested_*`: diagnósticos rolling con fit, selección, calibración y outer disjuntos dentro de cada fold.
- `report/tables/model_learning_curves.csv` y `model_generalization_gap.json`: curvas de aprendizaje y brecha con métricas no ponderadas comparables.
- `report/tables/final_explainability_stability.csv`: estabilidad SHAP en tres muestras y semillas; se interpretan bandas, no el orden exacto.
- `report/tables/design_embedding_*`: auditoría de representación con embeddings de entidad bajo protocolo idéntico; discriminación equivalente con 54 entradas y 6 042 parámetros frente a 169 y 12 993.
- `report/tables/design_capacity_*`: barrido de nueve arquitecturas (1 a 3 capas ocultas, 32 a 384 neuronas), curva de épocas del reajuste y curva de aprendizaje frente al tamaño de muestra.
- `data/processed/demo_cases.csv`: casos TN, FP, FN limítrofe y TP reales de 2023, más un clon sintético con código de vía no visto.
- `report/build_manifest.json`: frescura del PDF verificada por hashes de contenido.

## Incertidumbre

Los intervalos bootstrap remuestrean filas de la referencia con pipeline y predicciones congelados. No incluyen incertidumbre de entrenamiento o selección, dependencia temporal o espacial, consultas repetidas, generalización externa o futura, ni un intervalo de predicción individual. Los intervalos Wilson de la interfaz describen tasas agregadas, no incertidumbre por caso.
