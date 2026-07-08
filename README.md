# Clasificación de severidad de accidentes de tránsito en carreteras del Perú

Proyecto del curso **Redes Neuronales — Segunda Unidad**. El objetivo es construir una red neuronal para clasificación binaria de severidad de accidentes de tránsito en carreteras del Perú, con EDA, preprocesamiento, ingeniería de características, entrenamiento, evaluación, interfaz gráfica local con Streamlit, informe y anexos.

## Integrantes

- Rendo — líder técnico, entrenamiento canónico en macOS.
- Yimmy — EDA, GUI e informe bajo revisión técnica de Rendo.

## Fuente de datos

Dataset oficial: **Accidentes de tránsito en carreteras 2020-2021 – Sutran**.

- Fuente: Plataforma Nacional de Datos Abiertos.
- Recurso exacto: https://www.datosabiertos.gob.pe/dataset/accidentes-de-tr%C3%A1nsito-en-carreteras/resource/3398beff-8440-4343-a54d-0911d11dfcd5
- Fecha de verificación local en el repositorio: 2026-07-08.

Archivos versionados en `data/raw/`:

| Archivo | Tamaño | Observación |
|---|---:|---|
| `Accidentes de tránsito en carreteras-2020-2021-Sutran.csv` | 444600 bytes | CSV oficial; encoding detectado `latin-1`; separador real `;`. |
| `Formato_2_Diccionario_de_datos.docx` | 14887 bytes | Diccionario de datos asociado a la misma fuente oficial. |

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

## Ejecución prevista

### Notebooks

```bash
jupyter notebook notebooks/
```

Orden previsto:

1. `00_auditoria_dataset.ipynb`
2. `01_eda.ipynb`
3. `02_prep_features.ipynb`
4. `03_model.ipynb`

### Entrenamiento

El modelo canónico se entrena una sola vez en la máquina de Rendo. Los artefactos esperados se guardan en `models/`.

### Interfaz Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Informe LaTeX

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
```

## Bitácora de decisiones

| Fecha | Bloque | Situación | Decisión | Contingencia aplicada / Justificación |
|---|---|---|---|---|
| 2026-07-08 | A | Repositorio creado por Rendo | Se continúa desde verificación de estructura, entorno y permisos; no desde creación del repo | Estado confirmado por Rendo |
| 2026-07-08 | A/B | `Plan.md`, dataset y diccionario de datos ya subidos | Se verifica ubicación, fuente y reproducibilidad en README antes de procesar datos | Estado confirmado por Rendo |
| 2026-07-08 | B/H | Fuente exacta del dataset confirmada | Usar recurso oficial `3398beff-8440-4343-a54d-0911d11dfcd5` de Datos Abiertos; citarlo en README y `report/bib/referencias.bib` | Confirmado por Rendo + ficha oficial |
| 2026-07-08 | A | Python 3.12 no estaba disponible localmente como `python3.12` | Se instaló CPython 3.12.13 con `uv`, se creó `.venv`, se instalaron dependencias y se generó `requirements-macos.txt` | Verificación macOS: `import tensorflow, sklearn, pandas, pyarrow, shap, streamlit` ejecutado correctamente |
| 2026-07-08 | B | N registrado = 8117 tras saneamiento base | Arquitectura A; split posterior 70/15/15 | C2: 3000 ≤ N < 10000 |
| 2026-07-08 | B | Clase mortal = 951/8117 = 11.7162% | Balanceo posterior: `class_weight` solamente | C5: 5% ≤ clase MORTAL ≤ 25% |
| 2026-07-08 | B | CSV auditado y saneado | Se generó `data/processed/base_limpia.parquet` localmente y tablas de auditoría en `report/tables/` | Encoding `latin-1`, separador `;`, 35 duplicados eliminados, 3 targets NaN eliminados |
| 2026-07-08 | D | Split estratificado creado | Train/Val/Test = 5681/1218/1218; clase mortal 11.7057%/11.7406%/11.7406% | Diferencia < 1 punto porcentual entre subconjuntos |
| 2026-07-08 | D | Contrato de features congelado | Se generaron 72 features, `models/feature_list.json`, `models/scaler.pkl`, `models/encoders.pkl` y `tab02_feature_contract.csv` | Estadísticos fit solo con train; `preparar_entrada()` verificado |
| 2026-07-08 | D | Casos demo separados del test final | Se generó `data/processed/demo_cases.csv` desde validación y casos controlados | Incluye caso base, atropello, nocturno, código no visto y PUNO |
| 2026-07-08 | E | Grid cerrado de 6 corridas ejecutado | Mejor modelo: `R5_one_hidden_layer`; se guardó `models/severidad_nn.keras` | Selección por F1-mortal en validación; test no usado |
| 2026-07-08 | E | Umbral calibrado en validación | `threshold=0.5`; recall-mortal=0.5315; F1-mortal=0.3077 | Regla: máximo recall con F1 ≥ 90% del mejor F1 de validación |
| 2026-07-08 | E | Baselines comparados | Dummy, Regresión Logística balanceada y Random Forest balanceado registrados en `tab03` | La red supera al Dummy y queda por encima de los baselines en F1-mortal de validación |
| 2026-07-08 | F | Test final evaluado una sola vez | MLP: F1-mortal=0.2941, recall-mortal=0.4895, PR-AUC=0.2890, ROC-AUC=0.6840, accuracy=0.7241 | `test_evaluations=1`; umbral 0.5 congelado desde validación |
| 2026-07-08 | F | Falsos negativos analizados | 73 accidentes mortales quedaron bajo el umbral; se guardaron 5 ejemplos en `tab05_false_negatives_examples.csv` | Error de mayor costo: subestimación de accidente mortal |
| 2026-07-08 | F | SHAP y calibración generados | Top-5: `modalidad_despiste`, `mes`, `modalidad_atropello`, `nocturno`, `hora_cos`; se guardó curva de calibración | SHAP con muestra de test hasta 1000 filas por C14 preventivo |

