# Plan.md — Plan de Ejecución Cerrado
## Red Neuronal para Clasificación de Severidad de Accidentes de Tránsito en Carreteras del Perú
> [!IMPORTANT]
> ## BLOQUE DE INTEGRIDAD — DOCUMENTO CONGELADO
>
> Este `Plan.md` queda declarado como **fuente de verdad del proyecto**.
>
> **No debe ser editado, reescrito, resumido, reorganizado ni “mejorado” por ninguna IA, herramienta automática, asistente, agente o modelo generativo sin autorización explícita de Rendo.**
>
> Cualquier modificación futura debe cumplir obligatoriamente estas condiciones:
>
> 1. Ser solicitada o aprobada explícitamente por **Rendo**.
> 2. Registrar el cambio en la **Bitácora de Decisiones** del documento.
> 3. No alterar decisiones ya cerradas sin justificación técnica.
> 4. No reemplazar este plan por una versión genérica.
>
> Si una IA o herramienta automática vuelve a procesar este archivo, debe tratarlo como **documento congelado y verdad operativa del proyecto**, no como borrador editable.
>
> **Estado:** congelado como plan maestro.

> **Curso:** Redes Neuronales — Segunda Unidad
> **Dataset:** *Accidentes de tránsito en carreteras 2020-2021 – [Sutran]* — PNP + CGM/SUTRAN, Plataforma Nacional de Datos Abiertos. Recurso exacto usado por el equipo: `https://www.datosabiertos.gob.pe/dataset/accidentes-de-tr%C3%A1nsito-en-carreteras/resource/3398beff-8440-4343-a54d-0911d11dfcd5`.
> **Consigna literal del docente:** *"Una Red Neuronal para predicción o clasificación, con un dataset de la región o a nivel nacional. Además realizar EDA, Preprocesamiento, Ingeniería de Características en el dataset antes de entrenar el modelo, y que tenga una interfaz gráfica. Y un Informe y Anexos."* → Este proyecto responde con: red neuronal de **clasificación** (§1.1), dataset oficial **a nivel nacional**, EDA (Bloque C), Preprocesamiento (Bloque D), Ingeniería de Características (Bloque D), interfaz gráfica (Bloque G), Informe (Bloque H) y Anexos (Bloque H).
> **Equipo:** 2 integrantes — **Rendo** (macOS 26, líder técnico: dueño o revisor de todas las partes críticas) y **Yimmy** (Windows). Roles, dependencias y división del trabajo en §5. Este documento vive en la raíz del repositorio GitHub del proyecto y es la fuente de verdad del plan.
> **Estado actual del repositorio:** Rendo ya creó el repositorio y ya subió `Plan.md`, el dataset y el diccionario de datos. Por tanto, el Bloque A ya no empieza desde cero: queda como bloque de **verificación, normalización de estructura, entorno y reproducibilidad** antes de ejecutar el pipeline.
> **Plazo:** **2 días** (no semanas). Cronograma comprimido por sesiones en §12. Disponibilidad confirmada: ~8 h/día por persona (~32 horas-persona en total), suficiente para el alcance completo sin recortes.
> **Interfaz gráfica:** **web local con Streamlit — confirmada como aceptable por el docente.** No hay contingencia de escritorio.
> **Naturaleza de este documento:** plan de ejecución **cerrado**. Cada bloque define ENTRADA → TAREAS → SALIDA → CRITERIO DE ACEPTACIÓN. Las decisiones están tomadas por adelantado, incluidas las **contingencias** (§2), para que ningún paso se invente a mitad de camino.
> **Regla del documento:** si durante la ejecución surge algo no contemplado aquí, se registra en la Bitácora de Decisiones (§11) con fecha y justificación — nunca se improvisa en silencio.

---

# PARTE I — FUNDAMENTOS Y DECISIONES

## 1. Definición del problema (sin ambigüedad)

### 1.1 ¿Clasificación o predicción? — aclaración conceptual obligatoria

La consigna dice "predicción o clasificación". En rigor técnico:

- **Clasificación:** la salida es una **categoría discreta** (mortal / no mortal). La red termina en una neurona con activación **sigmoide** (binario) o varias con **softmax** (multiclase), y se entrena con **entropía cruzada**.
- **Regresión (lo que coloquialmente llaman "predicción" numérica):** la salida es un **número continuo** (p. ej. cuántos fallecidos). La red termina en una neurona **lineal** y se entrena con **MSE/MAE**.

**Decisión del proyecto: CLASIFICACIÓN BINARIA.** Nuestra red *clasifica* la severidad y, al hacerlo, *predice* la probabilidad de que un accidente sea mortal — así se redacta en el informe para cubrir ambos términos de la consigna con precisión. No hacemos regresión de conteo de fallecidos porque: (a) el conteo tiene distribución muy sesgada con exceso de ceros, (b) la pregunta útil de seguridad vial es "¿mortal o no?", y (c) la clasificación permite toda la batería de evaluación estándar (matriz de confusión, F1, curvas ROC/PR) que demuestra dominio del curso.

### 1.2 Variable objetivo (target)

```
y = 1  si FALLECIDOS > 0   ("MORTAL")
y = 0  en caso contrario    ("NO MORTAL")
```

- Filas donde `FALLECIDOS` es NaN (los 3 casos "N.I." del diccionario) y no se puede inferir → **se eliminan** (no se imputa un target).
- **`FALLECIDOS` y `HERIDOS` quedan PROHIBIDAS como variables de entrada.** Definen el resultado posterior del accidente: usarlas sería fuga de datos (data leakage) y produciría un modelo trivial e indefendible. Esta prohibición se declara explícitamente en el informe (sección Preprocesamiento) como decisión metodológica.
- Nota de normalización: el diccionario puede referirse a `NUM_FALLECIDOS`/`NUM_HERIDOS`, pero en el CSV real las columnas aparecen como `FALLECIDOS`/`HERIDOS`. El código debe normalizar nombres y trabajar con los nombres reales del CSV.

### 1.3 Variables de entrada (features base, del diccionario oficial)

| Columna | Rol | Justificación |
|---|---|---|
| `FECHA` | Feature (derivar) | Estacionalidad, día de semana, feriados |
| `HORA` | Feature (derivar) | Nocturnidad, fatiga, visibilidad |
| `DEPARTAMENTO` | Feature categórica | Geografía del riesgo |
| `CODIGO_VIA` | Feature (derivar) | Vías con siniestralidad concentrada |
| `KILOMETRO` | Feature numérica | Tramos peligrosos |
| `MODALIDAD` | Feature categórica | Atropello/choque/despiste/volcadura difieren en letalidad |
| `FALLECIDOS` | SOLO target | Prohibida como entrada |
| `HERIDOS` | SOLO resultado posterior | Prohibida como entrada |
| `FECHA_CORTE` | DESCARTAR | Administrativa, sin señal |

### 1.4 Métricas de evaluación (definidas ANTES de entrenar — así trabaja un profesional)

El dataset estará **desbalanceado** (los accidentes mortales son minoría). Por lo tanto:

- **Métrica primaria de selección de modelo:** F1-score de la clase MORTAL.
- **Métricas de reporte obligatorio:** Precision y Recall de la clase MORTAL, PR-AUC, ROC-AUC, matriz de confusión completa, accuracy (solo como referencia, nunca como argumento).
- **Prohibido** argumentar calidad del modelo con accuracy sola: un modelo que responde siempre "no mortal" alcanza accuracy alta y es inútil.

**Interpretación de errores en ESTE dominio (va al informe, sección Resultados):**

| Error | Significado aquí | Costo real |
|---|---|---|
| **Falso Positivo (FP)** | El modelo dice "mortal" y no lo fue | Se sobreestima el riesgo de un escenario → recursos de fiscalización mal asignados. Costo moderado. |
| **Falso Negativo (FN)** | El modelo dice "no mortal" y sí lo fue | Se subestima un escenario letal → no se prioriza un tramo/horario peligroso. **Costo alto.** |

**Decisión derivada:** como el FN es más costoso, en la calibración del umbral (Bloque F) se privilegia el **recall de la clase mortal**, aceptando más FP. Esta asimetría de costos es exactamente el tipo de razonamiento que un jurado espera ver escrito.

### 1.5 Objetivos medibles del proyecto

- **OG:** Desarrollar una red neuronal de clasificación binaria de severidad de accidentes en carreteras del Perú, con pipeline completo (EDA → preprocesamiento → ingeniería de características → entrenamiento → evaluación → GUI), reproducible y documentado.
- **OE1:** Caracterizar la siniestralidad vial nacional mediante EDA (≥ 8 hallazgos interpretados).
- **OE2:** Construir ≥ 12 características derivadas de las 6 variables base, cada una justificada.
- **OE3:** Entrenar un MLP y compararlo contra tres baselines: `DummyClassifier` (clase mayoritaria), Regresión Logística y Random Forest. El MLP debe superar claramente al Dummy; si no supera a Regresión Logística o Random Forest, se reporta y discute honestamente (contingencia C13), porque en datos tabulares los árboles pueden competir o ganar.
- **OE4:** Identificar los 5 factores de mayor influencia en la mortalidad vía SHAP.
- **OE5:** Desplegar una GUI en Streamlit que reutilice el pipeline de entrenamiento sin discrepancias.

> Nota de honestidad calibrada: con 6 variables base, el techo de desempeño es limitado por naturaleza. **No se promete una cifra de F1/AUC por adelantado** — prometer números antes de ver los datos es mala práctica. Lo que se compromete es el criterio OE3 (superar baselines o justificar por qué no) y la calidad metodológica.

### 1.6 Separación de subconjuntos: entrenamiento, validación, test final y casos de interfaz

Para no confundir "probar la interfaz" con "evaluar el modelo", el proyecto manejará cuatro conceptos distintos:

| Conjunto / archivo | Uso permitido | Uso prohibido |
|---|---|---|
| **Train** | Ajustar pesos de la red, `class_weight`, scaler, encoders, imputaciones y frecuencias | Medir resultado final |
| **Validation** | Elegir hiperparámetros, aplicar `EarlyStopping`, elegir el mejor modelo y calibrar el **umbral de decisión** | Reportarlo como desempeño final |
| **Test final** | Evaluación final única del modelo ya cerrado | Elegir arquitectura, umbral, features o corregir el modelo después de verlo |
| **Casos GUI / demo** (`data/processed/demo_cases.csv`) | Probar que la interfaz predice sin romperse y que los cambios son coherentes | Entrenar, ajustar hiperparámetros o reportar métricas finales |

**Decisión:** no se crea un cuarto split grande porque el dataset es relativamente pequeño. Los casos de GUI se construyen desde el conjunto de validación y desde casos sintéticos controlados, nunca desde el test final. Estos casos sirven para comprobar la inferencia de la app, no para medir desempeño.

**Aclaración profesional:** en este trabajo se hará **calibración del umbral de decisión** usando validación. Esto no es lo mismo que **calibración probabilística**. La red entrega una probabilidad cruda; el umbral decide desde qué probabilidad se etiqueta como MORTAL. Si el tiempo alcanza, se agregará una curva de calibración/reliability y Brier score como diagnóstico, pero no se hará una calibración probabilística adicional salvo que el informe lo justifique explícitamente.

---

## 2. Tabla de contingencias (decisiones tomadas HOY para no improvisar mañana)

| # | Situación posible | Decisión predefinida |
|---|---|---|
| C1 | `df.shape` < 3 000 filas | Concatenar todos los años disponibles del mismo dataset en el portal (homogenizando columnas). Si aun así N < 3 000: red mínima (1 capa oculta de 16), dropout 0.5, y validación cruzada estratificada 5-fold en lugar de un solo split para estabilizar las métricas. Se declara la limitación en el informe. |
| C2 | 3 000 ≤ N < 10 000 | Arquitectura A (§ Bloque E). Split 70/15/15. |
| C3 | N ≥ 10 000 | Arquitectura B (§ Bloque E). Split 70/15/15. |
| C4 | Clase MORTAL < 5% del total | `class_weight` + SMOTE (solo train) y comparar ambos; reportar el mejor por F1-mortal en validación. |
| C5 | 5% ≤ clase MORTAL ≤ 25% | Solo `class_weight`. SMOTE no se usa (innecesario). |
| C6 | Clase MORTAL > 25% | Sin balanceo especial; solo métricas por clase. |
| C7 | `HORA` viene como texto `HH:MM` | `pd.to_datetime(..., format="%H:%M").dt.hour` |
| C8 | `HORA` viene como entero `HHMM` o `HMM` | `hora = valor // 100`, validando rango 0–23; fuera de rango → NaN |
| C9 | `FECHA` no parsea como `%Y%m%d` | Inspeccionar 20 valores crudos, probar `%d/%m/%Y`; registrar el formato real en la Bitácora |
| C10 | Encoding falla con `latin-1` | Probar `windows-1252`, luego `utf-8-sig`; registrar el que funcione |
| C11 | `CODIGO_VIA` no tiene patrón de prefijo extraíble | Usar solo frequency encoding del código completo y omitir las features de prefijo (F4.8–F4.9) |
| C12 | La clase "solo daños" (F=0 y H=0) < 10% de filas | Se descarta la extensión de 3 clases; el proyecto queda binario (ya es la decisión por defecto) |
| C13 | El MLP NO supera a los baselines en F1-mortal | NO se oculta. Se reporta la tabla comparativa y se discute en el informe (sección Discusión): en datos tabulares pequeños los métodos de árboles suelen competir o ganar; el MLP cumple el requisito académico y el análisis explica por qué. Esto SUMA nota si está bien argumentado. |
| C14 | SHAP demasiado lento con todo el dataset | Muestreo de 1 000–2 000 filas del test para el análisis SHAP (`shap.sample`), documentado |
| C15 | Alguna columna con > 40% de NaN tras convertir N.I. | Se descarta la columna y se registra en la Bitácora (no se imputa masivamente) |
| C16 | Aparecen filas duplicadas exactas | Se eliminan (`drop_duplicates`) y se reporta la cantidad |
| C17 | `KILOMETRO` con valores absurdos (> longitud máxima de la vía o negativos) | Recorte a NaN e imputación por mediana de train; umbral: percentil 99.5 como tope de sanidad |
| C18 | Se agota el tiempo del Día 2 y algo del alcance no entra | Prioridad de recorte (lo último en caer): (1) intactos → pipeline de datos, red entrenada, evaluación en test, informe; (2) se simplifica antes que eliminarse → GUI a solo pestaña Predicción + Sobre el modelo (se posponen mapa y dashboard EDA a "trabajo futuro"); (3) el grid de 6 corridas se reduce a 3 (R1, R2, R3). Nunca se recorta: la regla anti-fuga, la evaluación única en test, ni los baselines. |
| C19 | Diferencias numéricas menores entre macOS y Windows al re-ejecutar | Esperado (implementaciones BLAS distintas por plataforma). Por eso el **modelo canónico se entrena UNA sola vez en la máquina designada (§4.3)** y el artefacto `.keras` se versiona en Git; el otro integrante lo consume, no lo re-entrena. Las cifras del informe salen exclusivamente de ese artefacto. |
| C20 | Conflicto de Git en un notebook (.ipynb) | No se fusiona a mano. Se respeta la regla de propiedad (§5.5): el dueño del notebook resuelve quedándose con su versión e incorpora los cambios del otro manualmente. Prevención: nadie edita un notebook ajeno. |
| C21 | Una librería falla en una de las dos plataformas al instalar | Se registra en Bitácora; se busca la versión inmediata anterior compatible en ambas y se fija ese pin para los dos. Prohibido que cada máquina use versiones distintas de la misma librería. |
| C22 | La ficha de datosabiertos no descarga, cambió de URL o solo contiene registros fatales | No se migra automáticamente a un dataset fatal-only, porque se perdería la clase negativa (`y=0`) y ya no sería clasificación mortal/no mortal. Primero se verifica que el recurso tenga accidentes con `FALLECIDOS = 0` y `> 0`. Si no existe clase negativa, se registra en Bitácora y se elige otro recurso oficial que conserve ambas clases antes de avanzar al Bloque B. |
| C23 | Se necesitan casos para probar la interfaz de predicción | No se usan filas del test final. Se crea `data/processed/demo_cases.csv` desde validación y casos sintéticos controlados. Debe incluir al menos: caso típico no mortal, caso con `ATROPELLO`, caso nocturno, caso con código de vía no visto y caso con departamento PUNO. Estos casos prueban coherencia de la app, no desempeño del modelo. |

---

## 3. Teoría de redes neuronales aplicada a ESTE proyecto (base del marco teórico del informe)

Cada concepto de abajo se redacta en el informe **conectado al proyecto**, no como teoría suelta. Este es el mapa concepto → aplicación:

| Concepto | Qué es (1 línea para el informe) | Dónde se aplica aquí |
|---|---|---|
| Neurona artificial | Combinación lineal de entradas + sesgo, pasada por una activación: `a = f(Wx + b)` | Unidad básica de cada capa del MLP |
| MLP (perceptrón multicapa) | Red *feed-forward* densa: capas de neuronas totalmente conectadas | Nuestra arquitectura (datos tabulares → MLP es el estándar) |
| Activación ReLU | `max(0, x)`; evita el desvanecimiento de gradiente y es barata | Capas ocultas |
| Activación Sigmoide | Comprime a (0,1); interpretable como probabilidad | Capa de salida (binaria) |
| Función de pérdida | Mide el error a minimizar | **Binary cross-entropy** (entropía cruzada binaria) — la pérdida correcta para clasificación binaria con salida sigmoide |
| Descenso de gradiente / Backpropagation | Cálculo del gradiente de la pérdida respecto a cada peso (regla de la cadena) y actualización de pesos en dirección contraria | Motor del entrenamiento; se explica con el diagrama forward→loss→backward |
| Optimizador Adam | Descenso de gradiente con momento y tasa adaptativa por parámetro | Optimizador elegido, `learning_rate=1e-3` inicial |
| Época / batch | Época = una pasada por todo el train; batch = subconjunto por actualización | `batch_size=32` (N pequeño) o `64` (N grande), hasta 200 épocas con parada temprana |
| Sobreajuste (overfitting) | La red memoriza el train y falla en datos nuevos; se detecta cuando `val_loss` sube mientras `train_loss` baja | Riesgo ALTO aquí (dataset delgado) → por eso Dropout + EarlyStopping + red pequeña |
| Dropout | Apaga neuronas al azar en entrenamiento; obliga a representaciones redundantes | 0.3–0.5 entre capas ocultas |
| Batch Normalization | Normaliza activaciones por batch; estabiliza y acelera | Tras la primera capa densa |
| Early Stopping | Detiene el entrenamiento cuando la métrica de validación deja de mejorar | `patience=15`, `restore_best_weights=True`, monitor `val_pr_auc` |
| Escalado de entradas | Las redes convergen mal con features en escalas dispares | `StandardScaler` en numéricas (ajustado SOLO en train) |
| Codificación de categóricas | Las redes solo consumen números | One-Hot (baja cardinalidad) y frequency encoding (alta cardinalidad) |
| Desbalance de clases | La clase minoritaria domina el costo real pero no la pérdida | `class_weight` inverso a la frecuencia; opcionalmente SMOTE (C4) |
| Matriz de confusión | TP/TN/FP/FN | Tabla central de Resultados, con la lectura de costos de §1.4 |
| Precision / Recall / F1 | P = TP/(TP+FP); R = TP/(TP+FN); F1 = media armónica | Métricas por clase, foco en MORTAL |
| Curva ROC y AUC | TPR vs FPR variando el umbral | Reporte; menos informativa con fuerte desbalance |
| Curva Precision-Recall y PR-AUC | P vs R variando el umbral | **Métrica de monitoreo principal** (la correcta bajo desbalance) |
| Umbral de decisión | La sigmoide da probabilidad; clasificar exige un corte | Se calibra en validación privilegiando recall-mortal (§1.4) — NO se asume 0.5 |
| Calibración probabilística | Diagnóstico de si las probabilidades predichas se parecen a frecuencias reales observadas | Opcional: curva de calibración y Brier score en resultados; no confundir con calibración del umbral |
| Fuga de datos (leakage) | Información del futuro/objetivo se cuela en las entradas | Dos guardas: (1) `FALLECIDOS`/`HERIDOS` prohibidas, (2) todo fit de preprocesamiento solo en train |
| Reproducibilidad | Mismos datos + mismo código = mismos resultados | Semillas fijas (`numpy`, `tensorflow`, `random`, `PYTHONHASHSEED`), versiones en `requirements.txt` |
| Interpretabilidad (SHAP) | Contribución marginal de cada feature a cada predicción | Top-5 factores de riesgo (OE4) y explicación por caso en la GUI |

---

## 4. Stack tecnológico FIJADO (multiplataforma macOS + Windows)

> Contexto real del equipo: **Rendo trabaja en macOS 26 (Apple Silicon)** y **Yimmy trabaja en Windows**. Todo el stack está elegido para funcionar idéntico en ambas plataformas, sin componentes deprecados.

### 4.1 Lenguaje y librerías (con pines)

| Componente | Versión fijada | Justificación / nota de plataforma |
|---|---|---|
| Python | **3.12.x** | Versión con soporte garantizado por TODO el stack en ambas plataformas. NO usar 3.13 (soporte aún parcial en parte del ecosistema) ni <3.10. |
| TensorFlow | **>=2.18, <2.20** | Incluye **Keras 3** integrado (`from tensorflow import keras`). Para este proyecto se usará instalación CPU oficial con `pip install tensorflow` en ambas plataformas. En macOS/Apple Silicon no se dependerá de aceleración GPU ni de `tensorflow-metal`: puede existir como plugin, pero **no será requisito** porque añade variabilidad y no es necesario para un MLP tabular pequeño. En Windows se asume CPU (TensorFlow nativo dejó de incluir GPU desde 2.11); esto es suficiente para el alcance. Formato de guardado: **`.keras`** (nativo de Keras 3; no usar el legado `.h5`). |
| scikit-learn | >=1.6 | Split, baselines, métricas, `StandardScaler`, `compute_class_weight` |
| pandas | >=2.2 | Carga, limpieza. API estable 2.x |
| numpy | >=2.0 | Compatible con TF>=2.18 y shap>=0.46 |
| pyarrow | última | **Requerido** para leer/escribir Parquet con pandas |
| imbalanced-learn | >=0.13 | SMOTE (solo si aplica C4) |
| shap | >=0.46 | Primera línea con soporte de numpy 2.x |
| matplotlib / seaborn / plotly | >=3.9 / >=0.13 / >=5.24 | Figuras EDA y de resultados |
| streamlit | >=1.40 | GUI (§4.4) |
| holidays | última | Feriados de Perú (`holidays.country_holidays("PE")`) |
| joblib | última | Persistencia de scaler/encoders |

**`requirements.txt` del repo (pines mínimos, idéntico para ambos):**
```
python-version: 3.12 (documentado en README, no es línea de pip)
tensorflow>=2.18,<2.20
scikit-learn>=1.6
pandas>=2.2
numpy>=2.0
pyarrow
imbalanced-learn>=0.13
shap>=0.46
matplotlib>=3.9
seaborn>=0.13
plotly>=5.24
streamlit>=1.40
holidays
joblib
```
Tras instalar, **cada integrante** genera su congelado exacto y lo versiona como evidencia de reproducibilidad: `pip freeze > requirements-macos.txt` / `pip freeze > requirements-windows.txt` (van al Anexo A6/A7).

### 4.2 Reglas de código multiplataforma (obligatorias)

1. **Rutas SIEMPRE con `pathlib`** (`Path("data") / "raw"`), jamás strings con `\` o `/` hardcodeados.
2. **Sin rutas absolutas** en ningún script/notebook: todo relativo a la raíz del repo.
3. `.gitattributes` en la raíz con `* text=auto` (normaliza fines de línea LF/CRLF entre macOS y Windows).
4. Encoding SIEMPRE explícito en cada `read_csv`/`to_csv`.
5. Activación del entorno: macOS `source .venv/bin/activate`; Windows PowerShell `.venv\Scripts\Activate.ps1` (si PowerShell bloquea scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, una sola vez).

### 4.3 Determinismo entre plataformas — decisión canónica

Aun con semillas fijas y versiones idénticas, macOS (ARM) y Windows (x86) pueden producir **diferencias numéricas mínimas** (bibliotecas BLAS distintas). Decisión profesional para que las cifras del informe sean únicas e indiscutibles:

> **El modelo canónico se entrena UNA sola vez, en la máquina de Rendo (macOS).** El artefacto `models/severidad_nn.keras`, junto con `scaler.pkl`, `encoders.pkl`, `feature_list.json` y `threshold.json`, **se versiona en Git**. Yimmy consume esos artefactos (GUI, verificaciones), NO re-entrena. Toda métrica reportada en el informe proviene de ese único artefacto (contingencia C19).

### 4.4 Interfaz gráfica — decisión y respaldo

- **Decisión: Streamlit** (interfaz gráfica web que corre en local con `streamlit run`). Cumple la consigna: la indicación del docente dice "interfaz gráfica" sin especificar web o escritorio, y Streamlit produce una interfaz gráfica completa e interactiva. Ventajas para este proyecto: desarrollo rápido, idéntica en macOS y Windows, demo en vivo limpia.
- **Confirmado con el docente:** una interfaz gráfica web ejecutada localmente (Streamlit) es aceptable. No se requiere aplicación de escritorio. Diseño completo de la interfaz en §6.

### 4.5 LaTeX — herramienta de trabajo local

- **Decisión: LaTeX local**, no Overleaf. Ambos trabajarán el informe dentro del repositorio en `report/`, para que el PDF, las secciones, las figuras y el historial queden controlados por Git.
- **Distribución recomendada:** Rendo instala MacTeX o BasicTeX + `latexmk`; Yimmy instala MiKTeX o TeX Live. La instalación exacta usada por cada uno se registra en el README y en el Anexo A6 si hubo diferencias.
- **Editor recomendado:** VS Code + extensión LaTeX Workshop. No es obligatorio, pero reduce errores de compilación y permite compilar con un comando.
- **Estructura:** `report/main.tex` con `\input{sections/...}`. La carpeta `report/` concentra el informe y sus evidencias: `sections/`, `bib/`, `figures/`, `tables/` y `output/`. Cada sección vive en un archivo propio para evitar conflictos: `01_introduccion.tex`, `03_marco_teorico.tex`, `10_resultados.tex`, etc.
- **Compilación local estándar:** desde `report/`, ejecutar `latexmk -pdf -interaction=nonstopmode main.tex`. El PDF final se copia a `report/output/informe.pdf`.
- **Regla de trabajo:** nadie edita la misma sección `.tex` al mismo tiempo. Si una sección es crítica, Rendo la revisa antes de mergear. El PDF se recompila al cierre de cada sesión y, como mínimo, al final de cada día.

### 4.6 Control de versiones — GitHub

- Repositorio **privado** en GitHub. **Estado actual:** ya fue creado por **Rendo** y ya contiene `Plan.md`, el dataset y el diccionario de datos. Queda verificar que Yimmy tenga acceso como colaborador, que la estructura del repo coincida con este plan y que el README registre la fuente exacta del dataset.
- **Flujo de ramas:** `main` siempre funcional. Una rama por bloque (`bloque-b-datos`, `bloque-c-eda`, `bloque-g-gui`...). Al terminar un bloque → Pull Request → **revisión obligatoria del otro integrante** → merge. La revisión cruzada no es burocracia: garantiza que AMBOS entienden todo el proyecto (crítico para la defensa oral).
- **Propiedad de notebooks:** los `.ipynb` se fusionan mal. Regla: **cada notebook tiene un único dueño y solo él lo edita** (contingencia C20). Observaciones del otro → comentarios en el PR. Antes de commitear un notebook: "Restart & Run All" y guardar.
- **`.gitignore`:** `.venv/`, `__pycache__/`, `.DS_Store`, `data/processed/` (regenerable desde raw + código). **SÍ se versionan:** `data/raw/` (dataset oficial y diccionario de datos ya subidos por Rendo; mantener nombres originales cuando sea posible, con URL y fecha en README), `models/` (artefactos canónicos, §4.3), `report/figures/`, `report/tables/` y `report/main.tex` con sus secciones.

---

## 5. Organización del equipo (Rendo + Yimmy): roles, dependencias y paralelo

### 5.1 Roles y principio de control

| Integrante | Máquina | Rol | Bloques como Responsable |
|---|---|---|---|
| **Rendo** | macOS 26 | **Líder técnico / dueño del núcleo** | B (datos), D (preprocesamiento/features), E (red), F (evaluación) + entrena el modelo canónico |
| **Yimmy** | Windows | **Análisis y producto** | C (EDA), G (GUI) |

**Principio de control (lo pediste explícitamente):** Rendo es **dueño o revisor obligatorio de TODO lo crítico**. "Crítico" = todo lo que determina la corrección del resultado y las cifras del informe:

- **Rendo ejecuta directamente** el núcleo del modelo: construcción del target y regla anti-fuga (B11, D), split y preprocesamiento (D), arquitectura y entrenamiento (E), y la **evaluación única en test** (F). Estas partes no se delegan.
- **Rendo revisa y aprueba (merge del PR)** todo lo que hace Yimmy antes de que entre a `main`: el EDA (porque sus hallazgos H1–H10 alimentan las features), la GUI (porque consume el modelo) y cada sección del informe redactada por Yimmy.
- Yimmy **no** puede mergear a `main` sin aprobación de Rendo. Rendo sí puede mergear lo suyo, pero pide a Yimmy una revisión rápida para que ambos entiendan todo (defensa oral).

Regla transversal: nadie entrega nada que Rendo no haya leído.

### 5.2 Matriz Responsable / Revisor por bloque

| Bloque | Responsable | Revisor / Aprueba merge | Entregable que habilita al otro |
|---|---|---|---|
| A | Rendo (repo) + cada uno su entorno | — | Repo + `requirements.txt` |
| B | **Rendo** | Yimmy (lectura) | `base_limpia.parquet` + Bitácora (N, % mortal, formatos) |
| C | Yimmy | **Rendo (aprueba)** | `fig01–fig13` + hallazgos **H1–H10** |
| D | **Rendo** | Yimmy (lectura) | `models/{scaler,encoders,feature_list.json}` + parquets train/val/test |
| E | **Rendo** | Yimmy (lectura) | `severidad_nn.keras` + `threshold.json` + `tab03` parcial + `tab04` |
| F | **Rendo** | Yimmy (lectura) | `tab03/tab05/tab06/tab07` finales + `fig16–fig21` + Top-5 SHAP + calibración |
| G | Yimmy | **Rendo (aprueba)** | App funcional + capturas `gui01–gui04` |
| H | ambos (§5.4) | **Rendo aprueba todo** | PDF compilado |
| I | ambos | Rendo valida corrida limpia | Corrida limpia en las dos plataformas |

### 5.2b Matriz operativa detallada por entregable crítico

| Entregable crítico | Responsable de producir | Responsable de revisar | Bloquea a | Evidencia mínima en repo |
|---|---|---|---|---|
| `README.md` operativo + fuente exacta del dataset | Rendo | Yimmy verifica que puede reproducir | A, B, H | `README.md` con comandos y URL oficial |
| `requirements.txt` + congelados por plataforma | Rendo coordina; ambos generan | Rendo revisa diferencias | Todos | `requirements.txt`, `requirements-macos.txt`, `requirements-windows.txt` |
| Auditoría del CSV y diccionario | Rendo | Yimmy lectura | B, H5 | notebook `00_auditoria_dataset.ipynb` o sección inicial de B + tabla de N.I. |
| `base_limpia.parquet` | Rendo | Yimmy lectura | C, D | `data/processed/base_limpia.parquet` + Bitácora |
| Figuras EDA + H1–H10 | Yimmy | **Rendo aprueba** | D9, H6, GUI EDA | `report/figures/fig01..fig13.png` + texto H1–H10 |
| Contrato de features | **Rendo** | Yimmy prueba consumo | E, G | `feature_list.json`, `scaler.pkl`, `encoders.pkl` |
| Pruebas del contrato `preparar_entrada()` | Rendo | Yimmy ejecuta en Windows | G, I | `tests/test_preprocessing_contract.py` o celda equivalente documentada |
| Modelos comparativos | **Rendo** | Yimmy lectura | H10, GUI modelo | `report/tables/tab03_modelos.csv`, `report/tables/tab04_grid.csv`, artefactos `.keras` |
| `threshold.json` + tabla de umbrales | **Rendo** | Yimmy lectura | G predicción | `models/threshold.json`, `report/tables/tab_umbral_validacion.csv` |
| `demo_cases.csv` | Rendo define; Yimmy prueba en GUI | Rendo aprueba coherencia | G, defensa | `data/processed/demo_cases.csv`, `report/tables/tab06_demo_cases.csv` |
| App Streamlit | Yimmy | **Rendo aprueba** | H11, defensa | `app/streamlit_app.py` + `report/figures/gui01..gui04.png` |
| `report/main.tex` y PDF | Ambos según §5.4 | **Rendo aprueba PDF final** | Entrega | `report/main.tex`, `report/output/informe.pdf` |
| Guion de defensa 10 min | Ambos | Rendo valida parte técnica | Defensa | `docs/defensa_10min.md` |

**Conclusión sobre organización:** la tabla de §5.2 define responsables por bloque; esta tabla §5.2b define responsables por artefacto. Con ambas, la organización queda completa: quién produce, quién revisa, qué bloquea y qué evidencia debe aparecer en el repositorio.

### 5.3 Grafo de dependencias — qué espera a qué

```
A (repo) ──► B (datos, RENDO) ──┬──► C (EDA, YIMMY) ──► [H1–H10] ──► D9 (congelar features, RENDO)
                                │
                                └──► D1–D8 (transforms base, RENDO)  ← EN PARALELO con C
                                              │
                                    D completo ──► E (red, RENDO) ──► F (evaluación, RENDO)
                                              │                            │
                          [feature_list.json] └──► G2 (contrato GUI, YIMMY)│
                                                                           │
        G1 (esqueleto de UI con datos simulados, YIMMY) ◄── NO depende de nada: arranca el Día 1
                                                                           │
                          [modelo + threshold] ────────────────────► G cableado final (YIMMY)
```

**Dependencias duras (bloqueantes) — "no puedo avanzar sin":**
1. Yimmy no arranca C sin `base_limpia.parquet` (de Rendo, B).
2. Rendo no congela features (D9–D12) sin H1–H10 (de Yimmy, C). *D1–D8 sí avanzan en paralelo mientras Yimmy hace el EDA.*
3. Rendo no arranca E sin D completo.
4. Yimmy no cablea la GUI final sin `models/` + `threshold.json` (de Rendo, E). *Pero sí arma toda la carcasa antes (G1).*
5. Informe §6 requiere C; §9–10 requieren E–F; §11 requiere G.

**Trabajo SIN dependencias (para que nadie esté parado):**
- **Yimmy, desde la hora 0:** esqueleto `report/main.tex` local, informe §1–§3 (contenido ya especificado en §1 y §3 de este plan), y la carcasa completa de la GUI con datos simulados (G1) — las 4 pestañas maquetadas sin modelo real.
- **Rendo, si espera el EDA:** avanza D1–D8, y redacta §4, §7–§8 del informe conforme ejecuta.

### 5.4 División del informe LaTeX (cerrada)

| Sección | Redacta | Aprueba | Cuándo puede empezar |
|---|---|---|---|
| Carátula, 1 Introducción, 2 Objetivos | Yimmy | Rendo | Día 1 hora 0 (contenido en §1 del plan) |
| 3 Marco teórico | Yimmy | **Rendo** (es el núcleo técnico) | Día 1 hora 0 (mapa en §3 del plan) |
| 4 Metodología | Rendo | Yimmy | Día 1 tarde |
| 5 Dataset | Yimmy | Rendo | Al cerrar Bloque B |
| 6 EDA | Yimmy | Rendo | Al cerrar Bloque C |
| 7 Preprocesamiento y 8 Ingeniería de características | **Rendo** | Yimmy | Al cerrar Bloque D |
| 9 Modelo | **Rendo** | Yimmy | Al cerrar Bloque E |
| 10 Resultados | **Rendo** (números/figuras) + Yimmy (redacción lectura FP/FN) | **Rendo** | Al cerrar Bloque F |
| 11 Interfaz gráfica | Yimmy | Rendo | Al cerrar Bloque G |
| 12 Discusión y 13 Conclusiones | AMBOS (sesión conjunta) | — | Día 2 tarde |
| Resumen | AMBOS (siempre al final) | Rendo | Día 2 tarde |
| Referencias (.bib) | cada uno las de sus secciones | — | continuo |
| Anexos A1–A3, A5 | Yimmy | Rendo | continuo |
| Anexos A4, A6, A7 | Rendo | — | continuo |

Regla: **las secciones técnicas núcleo (3, 7, 8, 9, 10) las aprueba Rendo sí o sí**, aunque Yimmy ayude a redactar. Toda sección pasa por revisión cruzada antes de cerrarse.

### 5.5 Reglas de trabajo (adaptadas a 2 días, trabajando juntos/conectados)

1. **Sync al inicio y fin de cada sesión** (4 sesiones, §12): revisar el grafo de dependencias y confirmar quién entrega qué a quién en esa sesión.
2. Un PR por bloque; commits con prefijo `[BLOQUE X]`. **Rendo aprueba los PR de Yimmy antes del merge** (§5.1).
3. Los artefactos de `models/` y los parquet de `data/processed/` se comparten **solo vía Git** (nunca USB/WhatsApp): una única fuente de verdad.
4. La Bitácora (§11) es compartida y obligatoria: cada decisión de contingencia la escribe quien la tomó, en el momento.
5. Al ir a solo 2 días trabajando en simultáneo, si Yimmy se bloquea esperando un entregable de Rendo, pasa de inmediato al pool sin dependencias (§5.3): informe o carcasa de GUI.

---

## 6. Diseño de la interfaz gráfica (definido y aprobado por Rendo)

> Web local con Streamlit (confirmada por el docente). **4 pestañas** (`st.tabs`). Toda transformación de entrada pasa por la función compartida `preparar_entrada()` (§ Bloque G) — la GUI nunca recodifica a mano. Estética: paleta sobria (rojo para riesgo alto, ámbar medio, verde bajo), título y logo institucional, textos en español.

**Nota de experto sobre el tipo de gráfico (para el informe y para Yimmy):** como el modelo es de **clasificación** (¿este accidente es mortal, sí/no?), la salida natural es un **medidor de probabilidad (gauge)**, no un gráfico de líneas. El gráfico de líneas se usa para *pronóstico de una serie en el tiempo* (predecir el futuro de una variable continua), que no es nuestro caso. Sí incluimos una **línea temporal en el dashboard del EDA** (accidentes por mes), porque ahí el eje tiempo sí tiene sentido descriptivo. Confundir ambos es un error conceptual que un experto detecta al instante — por eso queda documentado.

### Pestaña 1 — "Predicción de severidad" (la principal)
**Entradas** (formulario; valores válidos tomados de `feature_list.json`, nunca escritos a mano):
- Departamento (`selectbox`), Modalidad (`selectbox`: atropello/choque/despiste/especial/volcadura), Fecha (`date_input`), Hora (`slider` 0–23), Código de vía (`text_input` con ayuda), Kilómetro (`number_input` ≥ 0).

**Al pulsar "Analizar riesgo" se muestra:**
1. **Medidor tipo gauge** (Plotly) con la **probabilidad de accidente mortal** (0–100%), con zonas de color: verde <umbral bajo, ámbar medio, rojo ≥ umbral (el umbral es el de `threshold.json`, no 0.5).
2. **Etiqueta grande** MORTAL / NO MORTAL según ese umbral, con la probabilidad en número.
3. **Explicación del caso:** prioridad 1 = explicación rápida basada en los **Top-5 factores globales** del modelo y en los valores ingresados; prioridad 2 = **SHAP local** si responde en menos de 2 s. Si SHAP local demora o rompe la demo, no se fuerza en la interfaz: queda para el informe/anexos y la GUI muestra la explicación simplificada. Esto evita que una herramienta de interpretabilidad pesada ponga en riesgo la exposición.
4. **Nota de alcance** al pie: "Estimación de factores de riesgo con fines académicos; no es un predictor determinista."

### Pestaña 2 — "Análisis exploratorio (EDA)"
Dashboard con los gráficos clave del Bloque C (imágenes ya generadas o replicadas con Plotly interactivo):
- Tasa de mortalidad por **modalidad** (barras) — el hallazgo más fuerte.
- Accidentes por **hora del día** y tasa de mortalidad por hora.
- **Serie temporal**: accidentes por mes (aquí sí, línea temporal).
- Distribución del target (barras) mostrando el desbalance.
Cada gráfico con 1–2 líneas de lectura.

### Pestaña 3 — "Riesgo por departamento"
- **Prioridad 1 obligatoria:** ranking interactivo por departamento (barras + tabla ordenable) con N.° de accidentes y tasa de mortalidad. Esto no depende de GeoJSON y debe funcionar sí o sí.
- **Prioridad 2 opcional:** mapa coroplético del Perú (Plotly + GeoJSON de departamentos) coloreado por tasa de mortalidad o número de accidentes.
- Al pasar el cursor o seleccionar una fila: departamento, N.° de accidentes y % mortal.
- *Regla de tiempo:* si el GeoJSON consume demasiado tiempo o falla, no se bloquea la entrega. Se deja el ranking por barras como versión final y se registra en Bitácora que el mapa quedó como mejora futura.

### Pestaña 4 — "Sobre el modelo"
- `model.summary()` renderizado (arquitectura y n.° de parámetros).
- **Tabla comparativa** `tab03`: MLP vs DummyClassifier vs Regresión Logística vs Random Forest (F1-mortal, recall-mortal, PR-AUC…).
- **Matriz de confusión** (fig16) y curvas ROC/PR (fig17–fig18).
- Ficha del dataset (fuente oficial, N, periodo) y la declaración de la regla anti-fuga.
- Créditos: Rendo y Yimmy, curso, universidad.

**Criterio de aceptación de la GUI** (además del Bloque G): las 4 pestañas cargan sin error; la pestaña 1 responde en < 2 s; al cambiar solo la modalidad de una categoría de baja letalidad a `ATROPELLO`, la probabilidad sube de forma coherente con el EDA; la explicación local se muestra solo si no ralentiza la demo; la pestaña departamental funciona como barras/ranking aunque el mapa coroplético no se active.

---

# PARTE II — BLOQUES DE EJECUCIÓN

> Formato de cada bloque: **ENTRADA** (qué necesitas tener) → **TAREAS** (numeradas, exactas) → **SALIDA** (artefactos concretos) → **CRITERIO DE ACEPTACIÓN** (cómo sabes que el bloque está terminado). No se avanza al bloque siguiente sin cumplir el criterio del actual.

---

## BLOQUE A — Verificación del repositorio, entorno y reproducibilidad

**ENTRADA:** repositorio ya creado por Rendo, con `Plan.md`, dataset y diccionario de datos ya subidos.

**TAREAS**
- A1. Verificar y, si falta algo, completar la estructura de carpetas:
```
proyecto-accidentes-peru/
├── data/
│   ├── raw/               # CSV oficial + diccionario de datos original
│   ├── processed/         # base_limpia, train/val/test y demo_cases.csv
│   └── geo/               # opcional: GeoJSON Perú/departamentos
├── notebooks/             # 00_auditoria_dataset, 01_eda, 02_prep_features, 03_model
├── src/                   # data_loader.py, features.py, preprocessing.py, model_utils.py, evaluate.py
├── app/                   # streamlit_app.py
├── models/                # severidad_nn.keras, scaler.pkl, encoders.pkl, feature_list.json, threshold.json
├── report/
│   ├── main.tex
│   ├── sections/
│   ├── bib/referencias.bib
│   ├── figures/           # fig01..figNN + gui01..gui04
│   ├── tables/            # tab01..tabNN en CSV/LaTeX
│   └── output/            # PDF final
├── tests/                 # pruebas mínimas del contrato de preprocessing/GUI
├── docs/                  # defensa_10min.md y notas de entrega
├── requirements.txt
└── README.md
```
- A2. **[Rendo]** Confirmar que el repositorio privado ya contiene `Plan.md`, el dataset y el diccionario de datos; invitar o verificar acceso de **Yimmy** como colaborador; añadir o corregir `.gitignore` y `.gitattributes` (`* text=auto`) si todavía no existen.
- A2b. **[Rendo]** Ordenar los archivos ya subidos: dataset oficial y diccionario en `data/raw/`. Registrar en `README.md`: nombre exacto del archivo, fuente/URL, fecha de descarga, peso y observación de que el diccionario proviene de la misma fuente oficial. La URL oficial que debe quedar escrita en README y en `report/bib/referencias.bib` es: `https://www.datosabiertos.gob.pe/dataset/accidentes-de-tr%C3%A1nsito-en-carreteras/resource/3398beff-8440-4343-a54d-0911d11dfcd5`.
- A2c. **[Rendo]** Crear o actualizar `README.md` operativo con: descripción del proyecto, integrantes, dataset oficial, fuente exacta, estructura de carpetas, instalación, ejecución de notebooks, entrenamiento, ejecución de Streamlit y compilación de `report/main.tex`.
- A2d. **[Rendo]** Crear `requirements.txt`, `.gitignore` y `.gitattributes`. Crear también `tests/` y `docs/defensa_10min.md` aunque al inicio estén vacíos.
- A3. **[AMBOS, cada uno en su máquina]** Entorno virtual con **Python 3.12** + instalación desde el `requirements.txt` versionado (pines de §4.1):
```bash
# macOS (Rendo)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements-macos.txt

# Windows PowerShell (Yimmy)
py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip freeze > requirements-windows.txt
```
(El `requirements.txt` incluye `pyarrow` — necesario para Parquet. Si una librería falla en una plataforma → contingencia C21.)
- A3b. Commits por bloque terminado, mensaje `[BLOQUE X] descripción`; flujo de ramas y PR según §4.6.
- A4. Fijar el bloque de semillas que se copiará al inicio de TODO notebook/script:
```python
import os, random, numpy as np, tensorflow as tf
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

**SALIDA:** repositorio GitHub ya operativo y normalizado, con estructura coherente, dataset y diccionario ubicados, `README.md` con fuente de datos, `requirements.txt`, `.gitignore`, `.gitattributes`, y los dos congelados por plataforma.
**CRITERIO DE ACEPTACIÓN:** en **AMBAS máquinas** corre sin error `python -c "import tensorflow, sklearn, pandas, pyarrow, shap, streamlit"`; los dos integrantes clonaron el repo y pueden hacer push; dataset y diccionario están localizables desde rutas relativas; `README.md` documenta la fuente; `requirements-macos.txt` y `requirements-windows.txt` están commiteados.

---

## BLOQUE B — Adquisición y saneamiento base de datos

**ENTRADA:** Bloque A terminado.

**TAREAS**
- B0. Crear `notebooks/00_auditoria_dataset.ipynb` o una sección inicial equivalente en el notebook de limpieza. Debe verificar: separador real (`;`), encoding usado, columnas originales, columnas normalizadas, N original, N final, faltantes `N.I.` por columna, rango temporal y distribución `FALLECIDOS=0` vs `FALLECIDOS>0`. Esta auditoría es la primera evidencia del informe y del README.
- B1. Usar el/los archivos del dataset y diccionario **ya subidos al repositorio** por Rendo. No redescargar salvo que se detecte corrupción, archivo incompleto o falta de trazabilidad. Verificar que estén en una ruta reproducible (`data/raw/`) y que el `README.md` tenga: URL exacta del recurso oficial (`3398beff-8440-4343-a54d-0911d11dfcd5`), fecha de descarga, nombre y peso de cada archivo.
- B1b. Verificar antes de avanzar que el recurso contiene accidentes con `FALLECIDOS = 0` y `FALLECIDOS > 0`. Si solo contiene siniestros fatales o la ficha no descarga, aplicar C22. No se cambia de dataset en silencio.
- B2. Cargar con manejo de encoding (contingencias C10):
```python
df = pd.read_csv(path, encoding="latin-1")   # fallback: windows-1252, utf-8-sig
```
- B3. **Registrar N:** `df.shape` → anotar en Bitácora. Aplicar contingencia C1/C2/C3.
- B4. Si hay varios archivos/años: homogenizar nombres de columnas (mayúsculas, sin tildes, sin espacios) y concatenar. Registrar el rango temporal final.
- B5. Normalizar texto: `str.strip().str.upper()` en `DEPARTAMENTO`, `MODALIDAD`, `CODIGO_VIA`.
- B6. Convertir **todo `"N.I."` (y variantes `"N.I"`, `"NI"`, espacios)** a `np.nan` en todas las columnas.
- B7. Parsear `FECHA` (contingencia C9): `pd.to_datetime(df["FECHA"].astype(str), format="%Y%m%d", errors="coerce")`. Reportar cuántas fechas no parsearon.
- B8. Parsear `HORA` (contingencias C7/C8): primero `df["HORA"].dropna().sample(20)` para ver el formato REAL, luego aplicar el parser que corresponda. Resultado: columna `hora_entera` ∈ [0, 23] o NaN.
- B9. Sanidad de `KILOMETRO` (contingencia C17) y de `FALLECIDOS`/`HERIDOS` (no negativos; tope percentil 99.9).
- B10. Eliminar duplicados exactos (C16); reportar cuántos.
- B11. **Construir el target:** `y = (FALLECIDOS > 0).astype(int)`. Eliminar filas con `FALLECIDOS` NaN. Registrar el % de clase mortal → aplicar contingencia C4/C5/C6.
- B12. Descartar `FECHA_CORTE`. Guardar `data/processed/base_limpia.parquet`.

**SALIDA:** `base_limpia.parquet`; celda-resumen con: N inicial, N final, duplicados eliminados, % NaN por columna, % clase mortal, rango temporal.
**CRITERIO DE ACEPTACIÓN:** el parquet carga sin warnings; `hora_entera` solo contiene valores 0–23 o NaN; el target no tiene NaN; el resumen está escrito en la Bitácora (§11).

---

## BLOQUE C — EDA (Análisis Exploratorio de Datos)

**ENTRADA:** `base_limpia.parquet`.

> Regla del bloque: **cada figura se guarda numerada** en `report/figures/` (`fig01_...png`, 150 dpi, título y ejes en español) y **cada figura tiene 2–4 líneas de interpretación escrita**. Figura sin interpretación = figura que no existe para el informe.

**TAREAS (cada una = 1 figura + interpretación, salvo indicación)**
- C1. `fig01` Distribución del target (barras + %). → cuantifica el desbalance.
- C2. `fig02` Serie mensual de accidentes (línea). → tendencia y estacionalidad.
- C3. `fig03` Accidentes por hora del día (barras 0–23) **y** `fig04` tasa de mortalidad por hora (línea). → nocturnidad.
- C4. `fig05` Accidentes por día de semana + tasa de mortalidad por día.
- C5. `fig06` Accidentes por `MODALIDAD` y `fig07` **tasa de mortalidad por modalidad** (este suele ser el hallazgo más fuerte del proyecto).
- C6. `fig08` Top-15 departamentos por accidentes y `fig09` por tasa de mortalidad (ranking distinto = hallazgo).
- C7. `fig10` Top-15 `CODIGO_VIA` por número de accidentes; si el patrón de prefijo existe, agregado por prefijo de ruta.
- C8. `fig11` Distribución de `KILOMETRO` (hist) para las 3 vías con más accidentes. → tramos calientes.
- C9. `fig12` Mapa de calor mes × hora (accidentes). → patrones combinados.
- C10. `fig13` Matriz de % de faltantes por columna. → transparencia de calidad de datos.
- C11. Tabla `tab01`: estadísticos descriptivos de todas las variables (irá a Anexos).
- C12. **Redactar 8–10 hallazgos numerados en prosa** (H1…H10), cada uno con referencia a su figura. Ejemplo del formato: "H3: La tasa de mortalidad de los atropellos (fig07) duplica la de los choques, pese a representar menos del X% de los accidentes; esto anticipa que MODALIDAD será una feature dominante."

**SALIDA:** notebook `01_eda.ipynb` completo, ≥ 13 figuras numeradas, tabla `tab01`, lista H1–H10.
**CRITERIO DE ACEPTACIÓN:** el notebook corre de inicio a fin con "Restart & Run All" sin errores; todas las figuras existen como PNG; los hallazgos están escritos (no "pendientes").

---

## BLOQUE D — Preprocesamiento e Ingeniería de Características

**ENTRADA:** `base_limpia.parquet` + hallazgos del EDA.

> **Orden inviolable: primero SPLIT, después todo fit.** Cualquier estadístico usado para transformar (mediana, frecuencias, medias, scaler) se calcula SOLO con el train. Este es el segundo guardián anti-fuga del proyecto.

**TAREAS**
- D1. Split estratificado y reproducible:
```python
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=SEED)
# resultado: 70% train / 15% val / 15% test
```
Verificar que el % de clase mortal es ≈ igual en los tres conjuntos (imprimirlo). Con el dataset real ya observado, se espera aproximadamente N=8 152 filas útiles y clase MORTAL ≈ 11.7%; por eso aplica C2 + C5.

- D1b. **Casos de prueba para interfaz, separados del test final:** crear `data/processed/demo_cases.csv` usando filas del conjunto de validación y casos sintéticos controlados. Columnas mínimas: `caso_id`, `descripcion`, `FECHA`, `HORA`, `DEPARTAMENTO`, `CODIGO_VIA`, `KILOMETRO`, `MODALIDAD`, `esperado_cualitativo`, `motivo`. Casos obligatorios:
  1. `demo_01_tipico_no_mortal`: caso diurno y modalidad de baja letalidad.
  2. `demo_02_atropello`: igual al anterior, pero cambiando solo `MODALIDAD=ATROPELLO`; debe subir la probabilidad respecto a `demo_01`.
  3. `demo_03_nocturno`: igual al caso base, pero en horario nocturno; debe cambiar la probabilidad de forma razonable.
  4. `demo_04_codigo_no_visto`: código de vía inventado/controlado (`PE-999X`); la app no debe romperse y debe mapear `via_freq=0`.
  5. `demo_05_puno`: caso con `DEPARTAMENTO=PUNO` para verificar que el encoding geográfico funciona.

- D2. **Features temporales** (de `FECHA`): `anio`, `mes` (1–12), `dia_semana` (0–6), `fin_de_semana` (bool), `feriado` (bool, con `holidays.country_holidays("PE")`), `quincena` (bool: día 14–16 y 29–2, proxy de pago/viajes — justificar en informe).
- D3. **Features horarias** (de `hora_entera`): codificación cíclica `hora_sin = sin(2π·h/24)`, `hora_cos = cos(2π·h/24)`; `nocturno` (bool: 20:00–05:59); `franja` (MADRUGADA 0–5 / MAÑANA 6–11 / TARDE 12–17 / NOCHE 18–23). *Por qué cíclica: para la red, 23 y 0 deben ser vecinos; con la hora cruda distan 23 unidades.* Filas con hora NaN: `hora_sin=hora_cos=0` + flag `hora_faltante=1` (la red aprende del faltante en vez de recibir una mentira imputada).
- D4. **Geográficas:** One-Hot de `DEPARTAMENTO` (≈25 columnas, aceptable); `region_natural` (COSTA/SIERRA/SELVA) mapeada con un diccionario fijo departamento→región escrito en `features.py` (se incluye completo en Anexos).
- D5. **De vía** (contingencia C11): `via_prefijo` (parte alfabética del código, p. ej. `PE`) One-Hot; `via_freq` = frecuencia relativa del código completo **calculada en train** y mapeada a val/test (códigos no vistos → 0).
- D6. `KILOMETRO`: imputar mediana de train + flag `km_faltante`; luego escalar.
- D7. `MODALIDAD`: imputar `"DESCONOCIDO"` + One-Hot (ATROPELLO, CHOQUE, DESPISTE, ESPECIAL, VOLCADURA, DESCONOCIDO).
- D8. Escalar numéricas continuas (`KILOMETRO`, `via_freq`, `anio`) con `StandardScaler` (fit en train). Las cíclicas ya están en [-1,1] y los booleanos/One-Hot no se escalan.
- D9. Congelar la **lista final de features en orden fijo** → `models/feature_list.json`. La GUI consumirá este archivo: el orden de columnas es contrato, no convención.
- D10. Persistir transformadores: `models/scaler.pkl`, `models/encoders.pkl` (incluye el mapa de frecuencias de vía y el diccionario región) con `joblib.dump`.
- D11. Guardar matrices procesadas: `data/processed/{X_train,X_val,X_test,y_train,y_val,y_test}.parquet` y `data/processed/demo_cases.csv`.
- D12. Tabla `tab02` para el informe: feature | origen | tipo | transformación | justificación (una fila por feature final).
- D13. Crear prueba mínima del contrato de preprocesamiento: `tests/test_preprocessing_contract.py` o celda equivalente documentada. Debe comprobar que `preparar_entrada()` devuelve exactamente el número de columnas de `feature_list.json`, no genera NaN, respeta el orden de features, acepta código de vía no visto (`via_freq=0`) y rechaza o corrige entradas inválidas como `KILOMETRO < 0`.

**SALIDA:** notebook `02_prep_features.ipynb`, 6 parquet, `demo_cases.csv`, 3 artefactos en `models/`, `tab02` y prueba del contrato de preprocessing.
**CRITERIO DE ACEPTACIÓN:** ≥ 12 features finales; ningún estadístico calculado fuera de train (revisar celda por celda); `feature_list.json` existe; el % de clase mortal difiere < 1 punto entre train/val/test; `demo_cases.csv` existe y no proviene del test final; la prueba de contrato de `preparar_entrada()` pasa en macOS y Windows.

---

## BLOQUE E — Baselines y Red Neuronal

**ENTRADA:** matrices procesadas del Bloque D.

**TAREAS**

- E1. **Baseline 0 — DummyClassifier** (`strategy="most_frequent"`). Evalúa el modelo tonto que siempre predice la clase mayoritaria. Sirve para demostrar por qué la accuracy engaña: puede ser alta aunque el recall-mortal sea 0.
- E2. **Baseline 1 — Regresión Logística** (`class_weight="balanced"`, `max_iter=1000`). Evaluar en val: F1-mortal, PR-AUC, recall-mortal.
- E3. **Baseline 2 — Random Forest** (`n_estimators=300`, `class_weight="balanced"`, `random_state=SEED`). Mismas métricas en val.
- E4. Registrar los tres baselines en la tabla comparativa `tab03` (se completa con la red en E8). *Por qué baselines primero: sin punto de comparación, ninguna métrica de la red significa nada.*

- E5. **Definir la arquitectura según N** (contingencias C1–C3):

**Arquitectura A (3 000 ≤ N < 10 000):**
```python
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
```

**Arquitectura B (N ≥ 10 000):**
```python
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
```

- E6. Compilación y pesos de clase:
```python
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                       keras.metrics.Recall(name="recall"),
                       keras.metrics.Precision(name="precision")])

from sklearn.utils.class_weight import compute_class_weight
w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
class_weight = {0: w[0], 1: w[1]}
```

- E7. Entrenamiento:
```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                  patience=15, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("models/severidad_nn.keras",
                                    monitor="val_pr_auc", mode="max", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=200, batch_size=32, class_weight=class_weight,
                    callbacks=callbacks, verbose=2)
```
Guardar `fig14`: curvas de pérdida y PR-AUC train vs val (el diagnóstico de sobreajuste va al informe: si las curvas divergen, se comenta; si no, también).

- E8. **Búsqueda de hiperparámetros — grid cerrado (exactamente estas 6 corridas, ni una más ni una menos):**

| Corrida | Capas ocultas | Dropout | Learning rate |
|---|---|---|---|
| R1 (base) | según N (A o B) | base | 1e-3 |
| R2 | base | base | 1e-4 |
| R3 | base | +0.1 en todas | 1e-3 |
| R4 | mitad de neuronas por capa | base | 1e-3 |
| R5 | una capa oculta menos | base | 1e-3 |
| R6 | base | base | 1e-3, batch_size 64 |

Selección: **mayor F1-mortal en validación** (desempate: PR-AUC). Registrar las 6 en `tab04`. *Grid cerrado = nada de "probar una cosita más" a las 2 a.m.; si algo extraordinario lo justifica, va a la Bitácora.*

- E9. **Calibración del umbral de decisión en VALIDACIÓN** (nunca en test): barrer umbrales 0.05–0.95 (paso 0.05), graficar F1/precision/recall-mortal vs umbral (`fig15`), elegir el umbral según §1.4 (privilegiando recall-mortal con F1 razonable). Guardar `models/threshold.json`.
  - Importante: esto calibra el **corte de clasificación**, no re-entrena el modelo ni calibra probabilidades.
  - Opcional si el tiempo alcanza: calcular Brier score y curva de calibración en validación para diagnosticar si las probabilidades están bien calibradas. Si se observa mala calibración, se reporta como limitación; no se toca el test para corregirlo.
- E10. Si aplica C4: repetir E6 con SMOTE (solo train) y quedarse con la mejor variante por F1-mortal en val. Documentar ambas.

**SALIDA:** `03_model.ipynb`, `severidad_nn.keras`, `threshold.json`, `tab03` (parcial), `tab04`, `tab_umbral_validacion.csv`, `fig14`, `fig15`.
**CRITERIO DE ACEPTACIÓN:** las 6 corridas del grid están registradas; el modelo guardado corresponde a la mejor corrida; el umbral está persistido; **el test NO ha sido tocado todavía**.

---

## BLOQUE F — Evaluación final e interpretabilidad

**ENTRADA:** modelo final + umbral + `X_test/y_test` intactos.

**TAREAS**
- F1. **Una sola evaluación en test** (regla absoluta: el test se usa exactamente una vez; si se evalúa dos veces "para ver", deja de ser test):
  - `fig16` matriz de confusión (valores absolutos y normalizada).
  - `tab05` classification report completo (precision/recall/F1 por clase + soporte).
  - `fig17` curva ROC con AUC; `fig18` curva Precision-Recall con PR-AUC.
- F2. Completar `tab03`: DummyClassifier vs Regresión Logística vs Random Forest vs MLP en test (F1-mortal, recall-mortal, precision-mortal, PR-AUC, ROC-AUC, accuracy). Si el MLP no gana → contingencia C13 (se discute, no se esconde).
- F3. **Lectura de errores en el dominio** (párrafo obligatorio del informe): cuántos FN produjo el modelo, qué significan (accidentes mortales que el modelo habría subestimado), y cómo el umbral elegido balanceó ese costo. Analizar 3–5 falsos negativos concretos: ¿qué tenían en común?
- F4. **SHAP** (contingencia C14 si es lento): `fig19` summary plot (beeswarm) global; `fig20` bar plot de importancia media; identificar el **Top-5 de factores de riesgo** (cierra OE4) y contrastarlo con los hallazgos H1–H10 del EDA (¿la red "aprendió" lo que el EDA sugería? — esa coherencia es oro para la Discusión).
- F5. Prueba de reproducibilidad: reiniciar kernel, correr `03_model.ipynb` completo, verificar que las métricas coinciden (± ruido de GPU si aplica).
- F6. **Prueba de inferencia con casos GUI:** ejecutar el modelo sobre `demo_cases.csv` y guardar `report/tables/tab06_demo_cases.csv` con `caso_id`, `probabilidad_mortal`, `clasificacion`, `threshold`, `observacion`. Esta tabla no se usa como métrica; solo demuestra que la app predice y que los casos de control se comportan de forma coherente.
- F7. **Diagnóstico de calibración probabilística:** calcular Brier score y una curva de calibración (*reliability curve*) como diagnóstico final. Guardar `report/tables/tab07_calibracion.csv` y `report/figures/fig21_calibracion.png`. Esto no se usa para cambiar el modelo después del test; solo permite explicar con honestidad si la probabilidad mostrada en la GUI está bien calibrada o debe interpretarse como una estimación académica.

**SALIDA:** `fig16–fig21`, `tab03`, `tab05`, `tab06_demo_cases.csv` y `tab07_calibracion.csv` finales, párrafos de análisis de errores, calibración y coherencia EDA↔SHAP.
**CRITERIO DE ACEPTACIÓN:** el test se evaluó una sola vez; toda figura/tabla de resultados existe y está numerada; el Top-5 SHAP está escrito; los casos GUI se probaron sin usar el test final; la calibración se reportó como diagnóstico, no como excusa para re-ajustar el modelo.

---

## BLOQUE G — Interfaz gráfica (Streamlit)

**ENTRADA:** `severidad_nn.keras`, `scaler.pkl`, `encoders.pkl`, `feature_list.json`, `threshold.json`, `demo_cases.csv`.

**TAREAS**
- G1. `app/streamlit_app.py` con las **4 pestañas especificadas en §6** (Predicción / EDA / Mapa de riesgo / Sobre el modelo). Yimmy monta la carcasa de las 4 pestañas con datos simulados desde el Día 1; el cableado real espera el modelo (§5.3). La pestaña de predicción debe incluir un selector opcional “cargar caso de prueba” desde `demo_cases.csv`.
- G2. **Contrato de preprocesamiento:** la app construye el vector de entrada leyendo `feature_list.json` y aplicando `scaler.pkl`/`encoders.pkl` — **cero lógica de transformación duplicada a mano**. Función única `preparar_entrada(dict) -> np.array` en `src/preprocessing.py`, importada tanto por el notebook 02 como por la app (misma función = imposible divergir).
- G3. Pestaña 1: medidor gauge (Plotly) con la probabilidad y etiqueta según `threshold.json`; explicación simplificada con Top-5 globales y SHAP local solo si no supera 2 s. Pestaña 3: ranking por departamento obligatorio; mapa coroplético con GeoJSON en `data/geo/` solo como mejora opcional si el tiempo alcanza.
- G4. Manejo de entrada no vista: código de vía desconocido → `via_freq=0`; validar rangos (km ≥ 0); ningún input debe lanzar excepción visible.
- G5. Pruebas manuales (checklist): (a) las 4 pestañas cargan; (b) caso típico responde < 2 s; (c) al cambiar solo `MODALIDAD` de una modalidad de baja letalidad a `ATROPELLO`, la probabilidad sube de forma coherente con el EDA; (d) hora nocturna vs diurna produce cambios razonables; (e) código de vía no visto no rompe la app y usa `via_freq=0`; (f) la pestaña departamental funciona como barras/ranking aunque el mapa coroplético no se active.
- G6. Capturas de las 4 pestañas → `report/figures/gui01..gui04.png` (van a Anexos).

**SALIDA:** app funcional (`streamlit run app/streamlit_app.py`), 4 capturas.
**CRITERIO DE ACEPTACIÓN:** el checklist G5 pasa; la app usa `preparar_entrada` compartida (verificable en el código); las 4 capturas existen; **Rendo aprobó el PR**.

---

## BLOQUE H — Informe en LaTeX

**ENTRADA:** todas las figuras (`fig01–fig21`, `gui01–gui04`) y tablas (`tab01–tab07`) de los bloques previos.

> El informe se escribe en **LaTeX local** (`report/main.tex` con `\input{sections/...}`). La estructura de abajo es **cerrada**: cada sección lista su contenido obligatorio y las figuras/tablas que DEBE incluir. Nada queda "a ver qué pongo". La compilación estándar es `latexmk -pdf -interaction=nonstopmode main.tex` desde la carpeta `report/`.

**ESQUELETO Y CONTENIDO OBLIGATORIO**

| # | Sección | Contenido obligatorio | Figuras/Tablas |
|---|---|---|---|
| — | Carátula | Universidad, curso, título, integrantes, docente, fecha | — |
| — | Resumen | 150–200 palabras: problema, dataset, método, resultado principal (F1-mortal y top-3 factores), conclusión | — |
| 1 | Introducción | Siniestralidad vial en carreteras del Perú como problema público; pregunta de investigación: *¿qué circunstancias de un accidente en carretera se asocian a desenlace mortal y puede una red neuronal clasificarlas?* | — |
| 2 | Objetivos | OG + OE1–OE5 tal como están en §1.5 (medibles) | — |
| 3 | Marco teórico | 3.1 Seguridad vial (breve). 3.2 Redes neuronales: neurona, MLP, activaciones, pérdida (cross-entropy), backpropagation, Adam, sobreajuste y regularización (dropout, batchnorm, early stopping). 3.3 Clasificación vs regresión y por qué este problema es clasificación (§1.1). 3.4 Métricas bajo desbalance: matriz de confusión, precision/recall/F1, ROC, PR; costo asimétrico FP vs FN (§1.4). Todo redactado conectado al proyecto (mapa de §3 Parte I) | Diagrama propio de un MLP (TikZ o figura) |
| 4 | Metodología | CRISP-DM y correspondencia fase↔bloque (A–H); herramientas y versiones; semillas y reproducibilidad | Diagrama del pipeline |
| 5 | Dataset | Fuente oficial (SUTRAN/PNP, datosabiertos.gob.pe), licencia de datos abiertos, fecha de descarga, N, periodo, diccionario de datos (§1.3), tratamiento de "N.I." | tab01 |
| 6 | EDA | Hallazgos H1–H10 con sus figuras; párrafo de calidad de datos | fig01–fig13 (selección de 8–10; el resto a Anexos) |
| 7 | Preprocesamiento | Split estratificado 70/15/15; regla "fit solo en train"; **decisión anti-fuga: exclusión de FALLECIDOS/HERIDOS como entradas** (media página, es un pilar del trabajo); imputaciones con flags | — |
| 8 | Ingeniería de características | Tabla completa feature/origen/justificación; explicación de codificación cíclica de hora y del encoding de vía por frecuencia | tab02 |
| 9 | Modelo | Arquitectura final (diagrama + hiperparámetros), función de pérdida y optimizador con justificación, class_weight, callbacks, grid de 6 corridas, calibración del umbral de decisión; aclarar diferencia entre umbral y calibración probabilística | tab04, fig14, fig15 |
| 10 | Resultados | Métricas en test; comparación MLP vs baselines; análisis FP/FN con costos del dominio; SHAP y Top-5 factores; coherencia EDA↔SHAP; tabla de casos GUI como prueba funcional, no como métrica | tab03, tab05, tab06, tab07, fig16–fig21 |
| 11 | Interfaz gráfica | Arquitectura de la app, contrato de preprocesamiento compartido, capturas, protocolo de prueba con `demo_cases.csv` | gui01–gui04, tab06 |
| 12 | Discusión | Limitaciones honestas: 6 variables base → techo de desempeño; sesgo de registro policial; qué variables faltan (clima, vehículo, causa); si aplicó C13, por qué los árboles compiten en tabular; validez externa | — |
| 13 | Conclusiones | Una por objetivo (OE1–OE5): cumplido/cómo; conclusión general | — |
| 14 | Referencias | APA/IEEE en BibTeX: dataset (ficha oficial con fecha de consulta), documentación de Keras/scikit-learn/SHAP, texto base de NN (p. ej. Goodfellow et al., *Deep Learning*) | `report/bib/referencias.bib` |
| A | Anexos | A1 diccionario de datos completo; A2 tabla descriptivos; A3 figuras EDA no incluidas; A4 código clave (features, arquitectura, entrenamiento) en `listings`; A5 capturas GUI adicionales; A6 Bitácora de Decisiones; A7 enlace al repositorio | — |

**GUÍA DE TONO Y ESTILO (obligatoria — el lector es el catedrático, experto en redes neuronales)**

1. **Registro formal académico e impersonal.** Se redacta con pasiva refleja: "se entrenó el modelo", "se observa que", "se optó por". Nunca primera persona singular; "nosotros" solo si la plantilla institucional lo exige.
2. **Terminología precisa, en español, con el término inglés entre paréntesis la PRIMERA vez:** sobreajuste (*overfitting*), fuga de datos (*data leakage*), parada temprana (*early stopping*), entropía cruzada (*cross-entropy*). Después, solo el término en español. Toda sigla se define en su primera aparición (MLP, SHAP, PR-AUC…).
3. **Tiempos verbales:** metodología y resultados en pretérito ("se dividió el conjunto", "se obtuvo un F1 de…"); hechos generales de la teoría en presente ("la entropía cruzada penaliza…").
4. **Cero adjetivos promocionales.** Prohibido "excelente", "impresionante", "espectacular". Toda afirmación cuantitativa va anclada a su figura o tabla con `\ref{}`. Las limitaciones se declaran con la misma naturalidad que los logros — el lector es experto y detecta el maquillaje al instante.
5. **No explicar lo elemental de más.** El marco teórico no repite un libro de texto: demuestra dominio **conectando cada concepto con la decisión del proyecto que lo aplica** (el mapa de la §3 de este plan es exactamente eso). Un experto valora la conexión teoría→decisión, no la paráfrasis de definiciones.
6. **Notación matemática consistente** (`amsmath`): definir símbolos al usarlos (p. ej. $\hat{y} = \sigma(W_2\,f(W_1x + b_1) + b_2)$; $\mathcal{L}$ para la pérdida). La función de pérdida se escribe explícita:
$$\mathcal{L}(y,\hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}\left[w_1\, y_i \log \hat{y}_i + w_0\,(1-y_i)\log(1-\hat{y}_i)\right]$$
7. **Formato numérico uniforme:** métricas con 4 decimales (0.7312); separador decimal consistente en todo el documento (elegir punto o coma según norma de la universidad y declararlo en la Bitácora); porcentajes con 1 decimal.
8. **Español correcto:** el documento se pasa por corrector antes de compilar la versión final; tildes en mayúsculas incluidas (LaTeX con `babel[spanish]` las maneja bien).

**TAREAS**
- H1. Montar `report/main.tex` (documentclass `article` o plantilla de la universidad si existe; paquetes: `graphicx, booktabs, listings, hyperref, babel[spanish], amsmath, float`).
- H2. Redactar secciones en el ORDEN 5→6→7→8→9→10 (los resultados ya existen), luego 3–4, luego 1–2 y 12–13, al final Resumen (siempre se escribe último).
- H3. Toda figura con `\caption` interpretativo (qué muestra Y qué significa) y `\label` referenciado en el texto con `\ref`. Figura no referenciada en el texto = figura que sobra.
- H4. Compilar a PDF desde `report/` con `latexmk -pdf -interaction=nonstopmode main.tex`; revisión final de ortografía y numeración.

**SALIDA:** `report/output/informe.pdf`.
**CRITERIO DE ACEPTACIÓN:** compila sin errores; todas las secciones tienen su contenido obligatorio; ninguna figura/tabla sin referencia en el texto; los 5 objetivos tienen su conclusión.

---

## BLOQUE I — Cierre y ensayo de defensa

**TAREAS**
- I1. `README.md` final: descripción, cómo reproducir (comandos exactos), cómo lanzar la GUI y cómo compilar `report/main.tex`.
- I2. Corrida limpia total: clonar el repo en carpeta nueva → `pip install -r requirements.txt` → correr los 3 notebooks → lanzar la app. Si algo falla, es un bug de reproducibilidad: se arregla antes de entregar.
- I3. Crear `docs/defensa_10min.md` y realizar ensayo de defensa (10 min): 1 min problema → 2 min dataset y EDA (2 hallazgos fuertes) → 2 min decisiones anti-fuga y features → 2 min red y entrenamiento → 2 min resultados y SHAP → 1 min demo GUI en vivo.
- I4. Preparar respuestas a las 5 preguntas más probables del jurado: (1) *¿por qué no usaron HERIDOS como entrada?* → fuga, §1.2; (2) *¿por qué la accuracy no es más alta?* → techo por features, costo FN, §12 del informe; (3) *¿por qué una red y no solo Random Forest?* → requisito del curso + tabla comparativa honesta; (4) *¿cómo evitaron el sobreajuste?* → dropout, early stopping, curvas fig14; (5) *¿qué significa el umbral elegido?* → calibración por costo asimétrico, fig15.

**CRITERIO DE ACEPTACIÓN:** la corrida limpia funciona de punta a punta; el ensayo se hizo al menos una vez con tiempo tomado.

---

# PARTE III — CONTROL

## 10. Checklist maestro de errores prohibidos

- [ ] Usar `FALLECIDOS`/`HERIDOS` como entrada (fuga que define el target)
- [ ] Cualquier `fit` (scaler, encoder, imputador, SMOTE, frecuencias) fuera del train
- [ ] Tratar `"N.I."` como categoría real en vez de faltante
- [ ] Argumentar calidad con accuracy en clases desbalanceadas
- [ ] One-Hot masivo sobre `CODIGO_VIA`
- [ ] Hora sin codificación cíclica
- [ ] Red sobredimensionada para el N disponible
- [ ] Tocar el test más de una vez o usarlo para elegir hiperparámetros/umbral
- [ ] Usar filas del test final como casos de demo/prueba de la GUI
- [ ] Umbral 0.5 por defecto sin calibrar
- [ ] Semillas sin fijar / versiones sin congelar
- [ ] GUI con lógica de preprocesamiento duplicada (no compartida vía `preparar_entrada`)
- [ ] Presentar la red como superior sin la tabla de baselines
- [ ] Omitir el DummyClassifier como baseline de control
- [ ] Guardar tablas dentro de `report/figures/` en vez de `report/tables/`
- [ ] Mostrar probabilidad en la GUI sin aclarar calibración/alcance
- [ ] No probar `preparar_entrada()` contra `feature_list.json`
- [ ] Entregar PDF sin compilar localmente desde `report/main.tex`
- [ ] Figura sin interpretación o sin referencia en el texto
- [ ] Decisión tomada fuera de este plan sin registrarla en la Bitácora

## 11. Bitácora de Decisiones (plantilla — vive en `README.md` o `Anexo A6`)

| Fecha | Bloque | Situación | Decisión | Contingencia aplicada / Justificación |
|---|---|---|---|---|
| 2026-07-08 | A | Repositorio creado por Rendo | Se continúa desde verificación de estructura, entorno y permisos; no desde creación del repo | Estado confirmado por Rendo |
| 2026-07-08 | A/B | `Plan.md`, dataset y diccionario de datos ya subidos | Se verifica ubicación, fuente y reproducibilidad en README antes de procesar datos | Estado confirmado por Rendo |
| 2026-07-08 | B/H | Fuente exacta del dataset confirmada | Usar recurso oficial `3398beff-8440-4343-a54d-0911d11dfcd5` de Datos Abiertos; citarlo en README y `report/bib/referencias.bib` | Confirmado por Rendo + ficha oficial |
| | B | N registrado = ____ | Arquitectura __ | C1/C2/C3 |
| | B | % clase mortal = ____ | Balanceo: ____ | C4/C5/C6 |
| | B | Formato real de HORA = ____ | Parser: ____ | C7/C8 |
| | B | Encoding que funcionó = ____ | | C10 |
| | D | Patrón de CODIGO_VIA: ¿prefijo extraíble? __ | | C11 |
| | E | Mejor corrida del grid = R_ | Umbral = ____ | fig15 |
| | F | ¿MLP superó baselines? __ | | C13 si aplica |
| | A/G | Docente confirmó interfaz web local | Se usa Streamlit; no aplica contingencia de escritorio | Confirmado antes de ejecución |
| | B | Ficha oficial/dataset descargable y con clases 0/1: ____ | Se continúa / se aplica C22 | C22 si aplica |
| | G | SHAP local en GUI tarda > 2 s: ____ | Usar explicación simplificada + SHAP en informe/anexos | C14 + regla GUI §6 |
| | D/G | `demo_cases.csv` creado desde validación/sintéticos, no test: ____ | Casos GUI habilitados | C23 |
| | A | Librerías con problema de instalación por plataforma: ____ | Pin fijado: ____ | C21 |
| | H | Separador decimal elegido para el informe: ____ | | Guía de estilo, punto 7 |

## 12. Cronograma comprimido — 2 días / 4 sesiones (~32 horas-persona)

> Trabajan en simultáneo y conectados. Los puntos **S1–S5** son entregas por Git entre ambos; hasta que el artefacto no está en `main`, el dependiente trabaja en su pool sin dependencias (§5.3). Cada sesión abre y cierra con un sync (§5.5).

| Sesión | Carril Rendo (macOS) — núcleo | Carril Yimmy (Windows) — análisis/producto | Sincronización |
|---|---|---|---|
| **Día 1 – AM** (≈4 h) | A (verifica repo ya creado, estructura, dataset/diccionario, permisos y entorno macOS) → **Bloque B completo**: carga, N.I.→NaN, parseo FECHA/HORA, target, Bitácora (N, % mortal) → **D1–D8 + D1b** en paralelo | A (clona repo, verifica acceso y entorno Windows) → esqueleto `report/main.tex` local → informe **§1–§3** → **carcasa de la GUI (4 pestañas, datos simulados)** | **S1:** `base_limpia.parquet` + Bitácora en `main` → habilita el EDA de Yimmy |
| **Día 1 – PM** (≈4 h) | Recibe H1–H10 → **D9–D12** (congela features) → publica `feature_list.json` + scaler/encoders → arranca **Bloque E** | **Bloque C completo** (fig01–fig13 + **H1–H10**) → informe **§5–§6** → Rendo aprueba el PR del EDA | **S2:** H1–H10 → Rendo congela features. **S3:** `feature_list.json` + encoders → Yimmy cablea el contrato real en la GUI |
| **Día 2 – AM** (≈4 h) | **Bloque E** (grid R1–R6, curvas, umbral) → **Bloque F** (evaluación **única** en test + SHAP) → publica modelo, `tab03/tab05/tab06/tab07`, `fig16–fig21` → informe **§7–§9** | Con `feature_list.json`: pestañas EDA + Mapa de riesgo funcionando con datos reales; monta figuras/tablas en `report/main.tex` | **S4:** `severidad_nn.keras` + `threshold.json` + `demo_cases.csv` → GUI predice de verdad y puede probar casos controlados. **S5:** tablas/figuras de resultados (`report/tables/`, `report/figures/`) → habilita informe §10 |
| **Día 2 – PM** (≈4 h) | Informe **§4, §10**; aprueba PRs de Yimmy; **corrida limpia en macOS** (Bloque I) | **G cableado final** (gauge + SHAP + mapa) + pruebas G5 + capturas → informe **§11** → **corrida limpia en Windows** | Sesión conjunta: **§12–§13 + Resumen + ensayo de defensa** (I3–I4). La doble corrida limpia (macOS **y** Windows) es la prueba de reproducibilidad que se cita en el informe |

**Colchón de seguridad:** si al final del Día 2–PM el tiempo aprieta, se aplica la prioridad de recorte de la **contingencia C18** (nunca se recortan la regla anti-fuga, la evaluación única en test ni los baselines).

---

*Alcance declarado (informe y GUI): modelo académico de análisis de factores de riesgo de severidad; no es un sistema de predicción operativa de accidentes individuales. La honestidad sobre el alcance es parte de la calidad profesional del trabajo.*
