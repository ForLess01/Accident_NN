# Clasificación de alta letalidad en siniestros viales fatales del Perú

Proyecto del curso **Redes Neuronales — Segunda Unidad**. El objetivo es construir una red neuronal para clasificación binaria de alta letalidad (2+ fallecidos) en siniestros viales fatales del Perú, con EDA, preprocesamiento, ingeniería de características, entrenamiento, evaluación, calibración probabilística, interfaz gráfica local con Streamlit, informe y anexos.

## Integrantes

- Rendo — líder técnico, entrenamiento canónico en macOS.
- Yimmy — EDA, GUI e informe bajo revisión técnica de Rendo.

## Fuente de datos

Dataset oficial: **SINIESTROS DE TRANSITO FATALES 2021-2025 (PRELIMINAR)**.

- Autor oficial: Observatorio Nacional de Seguridad Vial (ONSV).
- Portal: https://www.onsv.gob.pe/datosabiertos
- Publicado por ONSV: 2026-02-27.
- Fecha de verificación local en el repositorio: 2026-07-09.
- El ONSV declara los datos de 2025 como preliminares.

Archivos versionados en `data/raw/`:

| Archivo | Observación |
|---|---|
| `BBDD_ONSV_SINIESTROS_FATALES_2021-2025.xlsx` | Base oficial ONSV “SINIESTROS DE TRANSITO FATALES 2021-2025 (PRELIMINAR)”; hoja `SINIESTROS`, encabezado en fila 5; 9,106 registros de datos. |
| `Accidentes de tránsito en carreteras-2020-2021-Sutran.csv` | Fuente de la primera iteración del proyecto (mortal/no-mortal); se conserva como evidencia histórica. |
| `Formato_2_Diccionario_de_datos.docx` | Diccionario de la fuente Sutran. |

## Definición del problema

Dado que ocurre un siniestro fatal, el modelo estima la probabilidad de que sea **multifatal** (2 o más fallecidos, 10.19% de los casos). El encuadre es condicional porque la base ONSV registra únicamente siniestros con al menos un fallecido. Columnas de resultado (lesionados, vehículos dañados) y de causa post-investigación quedan excluidas como features por fuga de datos.

## Estructura del repositorio

```text
.
├── Plan.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── geo/
├── notebooks/
├── src/
├── app/
├── models/
├── report/
│   ├── main.tex
│   ├── sections/
│   ├── bib/referencias.bib
│   ├── figures/
│   ├── tables/
│   └── output/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.12.x.
- Entorno virtual local por integrante.

### macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements-macos.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip freeze > requirements-windows.txt
```

## Ejecución

### Pipeline (scripts por bloque)

```bash
.venv/bin/python src/block_b_dataset_audit.py        # auditoría y limpieza
.venv/bin/python src/block_c_eda.py                  # figuras y hallazgos EDA
.venv/bin/python src/block_d_preprocessing.py        # split, features y artefactos
.venv/bin/python src/block_e_modeling.py             # baselines + grid MLP + umbral
.venv/bin/python src/block_f_evaluation.py           # evaluación única en test
.venv/bin/python src/block_f_posthoc_calibration.py  # calibración isotónica + bootstrap
.venv/bin/python src/block_g_app_check.py            # checklist funcional de la GUI
.venv/bin/python tests/test_preprocessing_contract.py
```

### Entrenamiento

El modelo canónico se entrena una sola vez en la máquina de Rendo. Los artefactos (`letalidad_nn.keras`, `calibrator.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_list.json`, `threshold.json`) quedan versionados en `models/`.

### Interfaz Streamlit

```bash
streamlit run app/streamlit_app.py
```

La GUI es de solo lectura: consume los artefactos versionados y no reentrena nada.

### Informe LaTeX

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
```

## Resultados principales (test, evaluación única)

| Métrica | Valor | IC 95% bootstrap |
|---|---:|---|
| ROC-AUC | 0.7474 | [0.7114, 0.7832] |
| Recall multifatal | 0.8849 | [0.8322, 0.9323] |
| F1 multifatal | 0.2746 | [0.2383, 0.3130] |
| PR-AUC (prevalencia 0.102) | 0.2038 | [0.1661, 0.2602] |
| Brier calibrado (isotónica) | 0.0872 | [0.0765, 0.0986] |

Top-5 SHAP: `clase_atropello`, `zona_rural`, `clase_despiste`, `clase_choque`, `zona_urbana`.

## Bitácora de decisiones

| Fecha | Bloque | Situación | Decisión | Contingencia aplicada / Justificación |
|---|---|---|---|---|
| 2026-07-08 | A–H | Primera iteración completa con fuente Sutran 2020-2021 (mortal/no-mortal) | Pipeline completo, informe y defensa; ROC-AUC 0.6840, empate con la logística | Diagnóstico: techo informacional por solo 6 variables base |
| 2026-07-09 | B | La única fuente peruana con variables pre-impacto a nivel registro (ONSV) es fatal-únicamente | Cambio de fuente a ONSV 2021-2025 y reformulación del target a alta letalidad (FALLECIDOS ≥ 2) | Evita sesgo de selección por mezcla de fuentes; decisión discutida en el informe |
| 2026-07-09 | B | Base auditada: 9,106 filas de datos, 2 fechas inválidas | N final 9,104; multifatal 928 (10.19%) | C2 (3000 ≤ N < 10000, split 70/15/15); C5 (class_weight, sin SMOTE) |
| 2026-07-09 | B | Columnas de resultado y causa disponibles en la base | Excluidas como features: LESIONADOS, VEHICULOS_DANADOS, CAUSA_*, señales (78% faltantes) | Guardia anti-fuga; verificada en el contrato de preprocesamiento |
| 2026-07-09 | D | Contrato de features congelado | 116 features; `feature_list.json`, `scaler.pkl`, `encoders.pkl`, `tab02_feature_contract.csv` | Estadísticos fit solo con train; `preparar_entrada()` verificado por test |
| 2026-07-09 | D | Split estratificado creado | Train/Val/Test = 6372/1366/1366; multifatal 10.20%/10.18%/10.18% | Diferencia < 1 punto porcentual |
| 2026-07-09 | E | Grid cerrado de 6 corridas ejecutado | Mejor modelo: `R5_one_hidden_layer`; guardado en `models/letalidad_nn.keras` | Selección por F1-multifatal en validación; test no usado |
| 2026-07-09 | E | Umbral calibrado en validación | `threshold=0.35`; recall 0.9209; F1 0.2744 | Regla: máximo recall con F1 ≥ 90% del mejor F1 de validación |
| 2026-07-09 | F | Test final evaluado una sola vez | MLP: recall 0.8849, F1 0.2746, PR-AUC 0.2038, ROC-AUC 0.7474 | `test_evaluations=1`; umbral 0.35 congelado desde validación |
| 2026-07-09 | F | La logística obtiene F1 nominalmente superior (0.3081) | Se reporta el empate estadístico con honestidad; el MLP gana en recall | C13; intervalos bootstrap traslapados |
| 2026-07-09 | F | Calibración post-hoc y CIs bootstrap | Calibrador isotónico ajustado en validación; `tab11`, `tab12`, `fig22` | Brier test baja de 0.2139 a 0.0872, mejor que el predictor de tasa base (0.0916) |
| 2026-07-09 | C | EDA regenerado para ONSV | `fig01`–`fig13`, `fig23` (scatter geográfico) y hallazgos H1–H11 | Hallazgo fuerte: rural 15.1% vs urbano 3.6%; lluvia 17.6% vs despejado 9.8% |
| 2026-07-09 | G | GUI adaptada al nuevo problema | Inputs de zona, red vial, clima, curvatura, perfil y superficie; gauge con probabilidad calibrada; mapa coroplético activo | Checklist G: 7/7 checks; verificada de punta a punta en navegador |
| 2026-07-09 | H/I | Informe y defensa reescritos | `report/main.pdf` (36 páginas) recompilado; `docs/defensa_10min.md` actualizado | Título, resumen, secciones y bitácora reflejan el problema reformulado |
