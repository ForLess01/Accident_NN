# Accident_NN - modelo definitivo de multifatalidad vial

Proyecto académico de redes neuronales que estima retrospectivamente la probabilidad de **dos o más fallecidos**, condicionada a un siniestro vial fatal registrado. El repositorio conserva una sola MLP definitiva, su evidencia reproducible, la interfaz Streamlit y el informe académico. La fuente no aporta timestamps por campo para demostrar operatividad al instante de notificación.

## Resultado principal

| Elemento | Valor definitivo |
|---|---|
| Fuente | ONSV, siniestros fatales del Perú 2021-2025 |
| Registros | 9 104 |
| Objetivo | `target_multifatal = 1` cuando `FALLECIDOS >= 2` |
| Entrenamiento | 2021-2022: 4 872 registros |
| Selección y calibración | 2023: 2 000 registros |
| Referencia histórica | 2024-2025: 2 232 registros |
| Bases companion (v2) | VEHICULOS (12 667 filas) y PERSONAS (25 412 filas), join 100% por código |
| Entrada | 26 campos crudos → 175 características procesadas |
| Arquitectura | `MLP_32_16`, ReLU, dropout 0.25, L2, semilla 314 |
| Calibración desplegada | Platt, seleccionada por Brier OOF en 2023 |
| Umbral calibrado | 0.30, seleccionado con predicciones OOF de 2023 |

La referencia 2024-2025 produjo PR-AUC **0.4416** [IC 95% 0.3785-0.5124], ROC-AUC **0.8841** [0.8613-0.9055] y F1 multifatal **0.5058** en la escala calibrada (Brier 0.0683, ECE 0.0169). El salto frente a v1 respalda fuertemente la hipótesis de un límite informacional, pero no la demuestra causalmente: cambian variables y configuración y la v2 es la segunda consulta declarada a esa referencia. La red supera a la regresión logística en ROC-AUC (Δ+0.026 [+0.012, +0.042]); frente al Random Forest no se detectó diferencia significativa, lo que no demuestra equivalencia.

La auditoría validation-only compara L2/dropout, una red, un ensemble de tres semillas y una multirrama. El ensemble mejora nominalmente PR-AUC en 2023, pero todos los IC pareados incluyen cero; por parsimonia se conserva **una sola MLP**. Una regla `n_personas >= 4`, elegida solo en 2023, alcanza F1 parecido en la referencia, pero la MLP la supera en PR-AUC (+0.0475 [0.0167, 0.0833]) y ROC-AUC (+0.0301 [0.0120, 0.0491]).

## Ejecución rápida en macOS

Requisitos verificados: macOS, Python **3.12** y un entorno virtual local.

```bash
cd /Users/rendoaltar/Developer/Accident_NN
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
python -m streamlit run app/streamlit_app.py
```

Abrí [http://localhost:8501](http://localhost:8501). La interfaz carga `models/final/`, verifica hashes y utiliza exactamente el mismo contrato de características del entrenamiento.

Si el entorno ya existe, el camino corto es:

```bash
cd /Users/rendoaltar/Developer/Accident_NN
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Cada sección tiene una URL compartible, por ejemplo `?section=estimar` o `?section=evidencia`.

## Verificación

```bash
.venv/bin/python scripts/check_release.py --local-content
```

`--local-content` es apropiado mientras los archivos definitivos todavía no estén confirmados en Git: valida contenido y reporta lo no versionado como advertencia. Antes de entregar desde un commit, ejecutá el gate estricto:

```bash
.venv/bin/python scripts/check_release.py
```

El gate ejecuta la suite directa, AppTest sobre las cinco secciones y los cinco escenarios, hashes del manifiesto, paridad de inferencia sobre 2 232 registros, guard de explicabilidad, escaneo de notebooks/imports, referencias obsoletas, vigencia del PDF y `git diff --check`. No reentrena ni inicia servidores persistentes.

Orden canónico de regeneración. La auditoría reentrena candidatos únicamente con 2021-2022, congela las decisiones en 2023 y recién entonces abre la referencia; el bundle reconstruye automáticamente el protocolo y los hashes de esa evidencia. Después se regeneran las figuras y el PDF:

```bash
.venv/bin/python src/validation_design_audit.py
.venv/bin/python src/final_model_bundle.py
.venv/bin/python src/final_evaluation_figures.py
.venv/bin/python src/final_explainability.py
cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

El entrenamiento completo solo es necesario si se decide rehacer la búsqueda predeclarada:

```bash
.venv/bin/python src/block_e_modeling.py
```

No se debe ajustar arquitectura, calibrador o umbrales con 2024-2025: esas etiquetas ya fueron observadas y solo constituyen referencia histórica.

## Formulario académico

- Los campos requeridos comienzan vacíos; los opcionales usan `NO INFORMADO`.
- **Escenario de demostración** permite elegir entre cinco registros controlados y comparar la respuesta del modelo sin atribuir causalidad.
- La fecha se restringe al periodo documentado 2021-2025.
- La ubicación puede elegirse con un clic en un mapa OpenStreetMap acotado al Perú (con carreteras, calles y límites departamentales); el clic completa latitud, longitud y deduce el departamento con los mismos polígonos versionados de la validación. Los campos de escena (zona, tipo de vía, clima) no se deducen del mapa porque describen cómo el ONSV registró el siniestro. El mapa requiere internet para el fondo; sin conexión, los campos numéricos siguen operativos.
- Las coordenadas deben formar un par dentro del Perú y corresponder al departamento elegido.
- La sección de escena (v2) pide vehículos y personas involucradas: totales requeridos, desgloses opcionales con validación de coherencia, y edad media opcional. No se ingresa ningún desenlace por persona.
- El código de vía ofrece búsqueda sobre códigos conocidos y advierte cuando recibe uno nuevo con formato válido.
- Los comparadores de subgrupo con soporte menor que 30 se enmascaran.

## Estructura canónica

```text
app/streamlit_app.py              interfaz profesional
src/block_b_dataset_audit.py      limpieza reproducible de la fuente ONSV
src/block_c_eda.py                análisis exploratorio
src/model_protocol.py             corte cronológico y 175 características
src/final_paired_comparison.py    bootstrap pareado vs logística y Random Forest
src/validation_design_audit.py    ablación, una-vs-varias redes y regla n_personas
src/block_e_modeling.py           búsqueda y selección de la MLP
src/final_model_bundle.py         calibración, evaluación, hashes e inferencia
src/final_evaluation_figures.py   figuras de selección y comparación
src/final_explainability.py       Gradient SHAP global sin usar 2024-2025
models/final/                     único bundle de inferencia
report/                           informe, figuras y tablas definitivas
docs/defensa_10min.md             guion de sustentación
```

## Interpretación académica

- El target es la multifatalidad entre siniestros fatales registrados por el ONSV.
- PR-AUC es central porque la clase multifatal representa cerca del 10 %.
- La calibración mejora la lectura probabilística, pero no crea capacidad discriminativa.
- Gradient SHAP describe asociaciones del modelo, no causalidad.
- Persisten variables ausentes como velocidad, protección y exposición; su mejora potencial requiere evidencia nueva.

## Errores comunes

| Mensaje o síntoma | Solución |
|---|---|
| `python3.12: command not found` | Instalá Python 3.12 y repetí la creación de `.venv`. |
| `No module named streamlit` | Activá `.venv` e instalá `requirements-macos.txt`. |
| Error al abrir el Excel ONSV | Confirmá `openpyxl==3.1.5` y `et_xmlfile==2.0.0` con `python -m pip show openpyxl et_xmlfile`. |
| Hash canónico inválido | Restaurá el archivo indicado; la app protege la correspondencia entre modelo y evidencia. |
| Puerto 8501 ocupado | Ejecutá `python -m streamlit run app/streamlit_app.py --server.port 8502`. |
| El gate estricto informa archivos no versionados | Confirmá los archivos definitivos en Git antes de la entrega; durante desarrollo usá `--local-content`. |

## Informe

El documento final está en [`report/main.pdf`](report/main.pdf). La especificación y el estado definitivo del proyecto están en [`Plan.md`](Plan.md). La reproducibilidad de un clon debe declararse únicamente después de que todos los artefactos canónicos requeridos estén versionados; el gate estricto lo comprueba.
