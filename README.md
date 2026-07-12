# Accident_NN - modelo definitivo de multifatalidad vial

Proyecto académico de redes neuronales que estima la probabilidad de **dos o más fallecidos**, condicionada a que el registro corresponde a un siniestro vial fatal ya notificado. El repositorio conserva una sola implementación: la MLP definitiva, su evidencia reproducible, la interfaz Streamlit y el informe académico.

## Resultado principal

| Elemento | Valor definitivo |
|---|---|
| Fuente | ONSV, siniestros fatales del Perú 2021-2025 |
| Registros | 9 104 |
| Objetivo | `target_multifatal = 1` cuando `FALLECIDOS >= 2` |
| Entrenamiento | 2021-2022: 4 872 registros |
| Selección y calibración | 2023: 2 000 registros |
| Referencia histórica | 2024-2025: 2 232 registros |
| Entrada | 162 características sin variables de resultado |
| Arquitectura | `MLP_64_32`, ReLU, dropout 0.35, L2, semilla 314 |
| Calibración desplegada | Platt, seleccionada por Brier OOF en 2023 |
| Umbral calibrado | 0.20, seleccionado con predicciones OOF de 2023 |

La referencia 2024-2025 produjo PR-AUC **0.2249**, ROC-AUC **0.7482** y F1 multifatal **0.2958** en la escala calibrada. La MLP supera nominalmente a los baselines en métricas de ranking; la regresión logística conserva mayor F1 (**0.3183**). Por rigor, el proyecto **no afirma superioridad universal** de la red.

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
./scripts/check_release.py --local-content
```

`--local-content` es apropiado mientras los archivos definitivos todavía no estén confirmados en Git: valida contenido y reporta lo no versionado como advertencia. Antes de entregar desde un commit, ejecutá el gate estricto:

```bash
./scripts/check_release.py
```

El gate ejecuta las cuatro pruebas directas, AppTest, hashes del manifiesto, paridad de inferencia sobre 2 232 registros, guard de explicabilidad, escaneo de notebooks/imports, referencias obsoletas, vigencia del PDF y `git diff --check`. No reentrena ni inicia servidores persistentes.

Regeneración de evidencia sin reentrenar la red:

```bash
.venv/bin/python src/final_model_bundle.py
.venv/bin/python src/final_evaluation_figures.py
.venv/bin/python src/final_explainability.py
```

El entrenamiento completo solo es necesario si se decide rehacer la búsqueda predeclarada:

```bash
.venv/bin/python src/block_e_modeling.py
```

No se debe ajustar arquitectura, calibrador o umbrales con 2024-2025: esas etiquetas ya fueron observadas y solo constituyen referencia histórica.

## Formulario académico

- Los campos requeridos comienzan vacíos; los opcionales usan `NO INFORMADO`.
- **Cargar caso de demostración** completa explícitamente un registro real de validación.
- La fecha se restringe al periodo documentado 2021-2025.
- Las coordenadas deben formar un par dentro del Perú y corresponder al departamento elegido.
- El código de vía ofrece búsqueda sobre códigos conocidos y advierte cuando recibe uno nuevo con formato válido.
- Los comparadores de subgrupo con soporte menor que 30 se enmascaran.

## Estructura canónica

```text
app/streamlit_app.py              interfaz profesional
src/block_b_dataset_audit.py      limpieza reproducible de la fuente ONSV
src/block_c_eda.py                análisis exploratorio
src/model_protocol.py             corte cronológico y 162 características
src/block_e_modeling.py           búsqueda y selección de la MLP
src/final_model_bundle.py         calibración, evaluación, hashes e inferencia
src/final_evaluation_figures.py   figuras de selección y comparación
src/final_explainability.py       Gradient SHAP global sin usar 2024-2025
models/final/                     único bundle de inferencia
report/                           informe, figuras y tablas definitivas
docs/defensa_10min.md             guion de sustentación
```

## Interpretación académica

- El target es la multifatalidad entre siniestros fatales notificados por el ONSV.
- PR-AUC es central porque la clase multifatal representa cerca del 10 %.
- La calibración mejora la lectura probabilística, pero no crea capacidad discriminativa.
- Gradient SHAP describe asociaciones del modelo, no causalidad.
- El principal límite es informacional: faltan velocidad, alcohol, ocupantes, cinturón/casco y características vehiculares.

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
