# Guion de Defensa Oral (10 minutos) — Accident\_NN

> Proyecto: Clasificación de severidad de accidentes de tránsito en carreteras del Perú.
> Equipo: **Rendo** (líder técnico, macOS) y **Yimmy** (Windows).
> Curso: Redes Neuronales — Segunda Unidad.

---

## Estructura de la defensa (10 minutos en total)

| Tiempo | Bloque | Punto fuerte |
|---|---|---|
| 0:00--1:00 | Problema | Por qué clasificación binaria, no regresión |
| 1:00--3:00 | Dataset y EDA | 2 hallazgos fuertes: H3 (volume ≠ severidad) y H5 (modalidad domina) |
| 3:00--5:00 | Anti-fuga + features | Las 2 guardias + codificación cíclica + freq de vía |
| 5:00--7:00 | Red + entrenamiento | Por qué grid cerrado, early stopping con PR-AUC, umbral 0.50 |
| 7:00--9:00 | Resultados + SHAP | MLP supera baselines + top-5 SHAP coherente con EDA |
| 9:00--10:00 | Demo GUI en vivo | 4 pestañas, gauge, caso atropello |

---

## 1. Problema (1 min)

> "La siniestralidad vial es un problema de salud pública en el Perú. Nuestra pregunta: ¿qué circunstancias de un accidente en carretera se asocian a desenlace mortal, y puede una red neuronal clasificarlas?"
>
> "Elegimos **clasificación binaria mortal/no-mortal**, no regresión. Razones técnicas: el conteo de fallecidos tiene exceso de ceros y una distribución muy sesgada; la pregunta útil de seguridad vial es **¿mortal o no?**; y la clasificación nos permite usar la batería completa de métricas para desbalance (F1, PR-AUC, matriz de confusión). Al clasificar, el modelo produce una probabilidad de accidente mortal — cubriendo así ambos términos de la consigna (predicción y clasificación) con precisión."

## 2. Dataset y EDA (2 min)

> "Usamos el recurso oficial Sutran 2020--2021 de datosabiertos.gob.pe: 8\,155 registros crudos, 8\,117 útiles tras eliminar 35 duplicados y 3 filas con target nulo. La clase mortal representa el **11.72 \%\**, que define el desbalance y el régimen metodológico."

> "**Dos hallazgos fuertes del EDA:**
>
> - **H3:** la hora con más accidentes (17:00) NO coincide con la hora de mayor tasa de mortalidad (01:00, 18.8 \%). Volumen y severidad son dimensiones distintas — esto justifica derivar `nocturno`, franja y codificación cíclica, no usar la hora cruda.
> - **H5:** DESPISTE es la modalidad más frecuente, pero ATROPELLO tiene la mayor tasa de mortalidad (42.9 \%). Anticipamos que MODALIDAD sería dominante — y SHAP lo confirmó."

## 3. Anti-fuga + features (2 min)

> "Tenemos **dos guardas anti-fuga**. Claridad metodológica absoluta:
>
> 1. `FALLECIDOS` y `HERIDOS` están **prohibidas como entradas**: `FALLECIDOS` define el target y `HERIDOS` es un resultado posterior. Usarlos sería *data leakage*. La auditoría del contrato de features lo verifica: **de 72 features finales, ninguna contiene esas columnas.**
> 2. **Primero split, después todo fit.** Cualquier estadístico (mediana, frecuencias, escalado) se calcula SOLO con train. Split estratificado 70/15/15 con `random_state=42`, y verificamos que el \% de clase mortal difiere en menos de un punto entre los tres subconjuntos."

> "Sobre features: construimos **72 características** derivadas de 6 variables base. Las tres decisiones técnicas más importantes:
>
> - **Codificación cíclica de hora** (`sin`/`cos` $2\pi h / 24$): para la red, 23 y 0 deben ser vecinos; con la hora cruda distan 23 unidades.
> - **Frequency encoding del código de vía** (calculada en train, mapeo a 0 para vías no vistas): porque el código tiene alta cardinalidad y un One-Hot masivo no generalizaría — una mala práctica que evitamos.
> - **Flags de faltantes** (`hora_faltante`, `km_faltante`): la red aprende del faltante en lugar de recibir una mentira imputada."

## 4. Red + entrenamiento (2 min)

> "El modelo es un **MLP** — el estándar para datos tabulares. Elegimos la arquitectura según el tamaño del dataset: como N está entre 3.000 y 10.000, usamos dos capas ocultas inicialmente. Tras un **grid cerrado de 6 corridas definido ANTES de entrenar**, ganó la corrida R5: una sola capa oculta de 32 neuronas, dropout 0.4, batch normalization, F1-mortal en val 0.3077."

> "Detalles técnicos profesionales:
> - Pérdida: **entropía cruzada binaria** con `class_weight='balanced'` (sin SMOTE, porque el 11.7 \% está en el rango 5--25 \%).
> - Optimizador: **Adam**, LR 1e-3.
> - Early stopping monitoreando **PR-AUC** (no val_loss), porque es la métrica correcta bajo desbalance. Restore best weights.
> - **Calibración del umbral en validación**: se barrió de 0.05 a 0.95. Se eligió **0.50** porque maximiza F1 (0.3077) con *recall*-mortal 0.5315. Costo asimétrico: un **falso negativo** (subestimar un accidente mortal) es **más caro** que un falso positivo — definir el umbral por costo operacional queda como trabajo futuro."

## 5. Resultados + SHAP (2 min)

> "El **test se evaluó exactamente una vez** — regla absoluta. Resultados:
>
> | Modelo | F1-mortal | Recall | PR-AUC |
> |---|---|---|---|
> | Dummy | 0.00 | 0.00 | 0.117 |
> | Reg. Logística | 0.272 | 0.532 | 0.282 |
> | Random Forest | 0.210 | 0.161 | 0.227 |
> | **MLP** | **0.294** | 0.490 | **0.289** |
>
> La red **supera a los 3 baselines** en F1-mortal y PR-AUC, por un margen estrecho sobre la regresión logística. Eso es **C13**: en datos tabulares chicos, los métodos clásicos compiten o ganan. Lo declaramos con honestidad desde el Plan y eso suma credibilidad."

> "**Top-5 SHAP**: `modalidad_despiste`, `mes`, `modalidad_atropello`, `nocturno`, `hora_cos`. Dos modalidades en el top-3 y `nocturno`+`hora_cos` en el top-5 — **el modelo aprendió lo que el EDA anticipaba**. Esta coherencia es oro ante un jurado."

> "**Honestidad sobre limitaciones**: Brier 0.2017, calibración deficiente; 6 variables base imponen un techo acotado. La nota de alcance en la GUI declara que es una **estimación académica de factores de riesgo**, no un predictor operacional."

## 6. Demo GUI en vivo (1 min)

> "Cuatro pestañas en Streamlit, todas consumen el mismo `preparar_entrada()` (cero lógica duplicada). Demo en vivo:
> - Pestaña 1: cargo el caso `demo_01_tipico_no_mortal` → prob. **0.30**.
> - Cambio solo `MODALIDAD` a `ATROPELLO` (demo_02) → prob. **0.80**. Coherente con H5.
> - Pestaña 2: EDA interactivo con los hallazgos H1--H10.
> - Pestaña 3: ranking por departamento (LA LIBERTAD es top por *tasa*, LIMA por *volumen*).
> - Pestaña 4: comparación contra baselines + matriz de confusión + curvas ROC/PR."

---

## 5 preguntas más probables del jurado — guion de respuesta

### P1. ¿Por qué no usaron `HERIDOS` como entrada?
- Es un **resultado posterior** del accidente, no una circunstancia *previa*. Usarlo sería **data leakage**: el modelo aprendería la causa desde el efecto, un artefacto trivial e indefendible.
- Es el segundo guardián anti-fuga del proyecto y queda verificado en el contrato de features.

### P2. ¿Por qué la *accuracy* no es más alta?
- Dos razones: (1) **6 variables base** imponen un techo acotado — faltan variables como clima, vehículo, conductor, causa. (2) Elegimos **privilegiar *recall*-mortal** bajando implícitamente precisión por el **costo asimétrico**: un falso negativo (accidente mortal subestimado) es más grave que un falso positivo.
- La métrica correcta aquí es **F1-mortal** y **PR-AUC**, no *accuracy* — el *DummyClassifier* alcanza 88 \% de *accuracy* prediciendo siempre "no mortal" y es inútil.

### P3. ¿Por qué una red y no solo Random Forest?
- **Cumplimos el requisito del curso**: red neuronal entrenada. Y la comparamos **honestamente** contra 3 baselines.
- El MLP gana en F1-mortal y PR-AUC, aunque por margen estrecho. El Random Forest tiene *accuracy* más alta solo porque su *recall*-mortal es bajísimo (0.16) — pierde los casos letales.

### P4. ¿Cómo evitaron el sobreajuste?
- Tres mecanismos: (1) **Dropout** 0.4 entre capas. (2) **Batch normalization** tras la primera capa densa. (3) **Early stopping** con `patience=15` y `restore_best_weights` monitoreando PR-AUC de validación. Ver curvas fig14: train y val no divergen notablemente.

### P5. ¿Qué significa el umbral 0.50 elegido?
- Es la calibración del **corte de clasificación** (no calibración probabilística). Barriendo umbrales en validación, 0.50 ofrece el mejor F1-mortal (0.3077) con *recall* 0.53. diferenciación clave:
  - **Umbral**: desde qué probabilidad se etiqueta como "MORTAL".
  - **Calibración probabilística**: si las probabilidades predichas coinciden con frecuencias reales (Brier 0.2017 = deficiente).
- La política operacional del umbral ideal depende de costos externos y debe definirse antes de evaluar un nuevo holdout — trabajo futuro.

---

## 3 preguntas adicionales de respaldo

### P6. ¿Por qué codificación cíclica de la hora y no la hora cruda?
- Para la red, 23 y 0 deben ser vecinos (media noche). Con la hora cruda, distan 23 unidades — rompe la continuidad cíclica del día.
- `sin` y `cos` de $2\pi h/24$ preservan esa vecindad. SHAP lo recompensa: `hora_cos` aparece en el top-5.

### P7. ¿Qué pasa si la app recibe un código de vía que nunca vio?
- Nada se rompe: `via_freq` se mapea a 0 (frecuencia de vía calculada en train). El contrato de features lo garantiza. Es el caso `demo_04_codigo_no_visto` y se prueba en la GUI.

### P8. ¿Por qué SMOTE no se usó?
- Regla de contingencia del Plan: si la clase mortal está entre 5 \% y 25 \%, solo `class_weight` (innecesario SMOTE). Como tenemos 11.7 \%, aplicamos C5: `class_weight='balanced'` sin SMOTE. SMOTE se reservó solo si fuera < 5 \%.

---

## Notas para el ensayo

- **Ensayar en vivo** con cronómetro: cada bloque debe durar el tiempo asignado. Si en la práctica se va a 11 min, recortar la demo (mostrar solo pestaña 1 con el caso atropello).
- **Tener abiertas las 4 pestañas** de la app antes de empezar, para no perder tiempo en la demo.
- **Tener el PDF del informe abierto** en la página de SHAP (Sección 10) por si preguntan top-5 con valores.
- **Mantener tono formal académico**: pasiva refleja ("se entrenó", "se observó"), sin primera persona.
- **Cerrar con una frase de alcance**: "El modelo es una **estimación académica de factores de riesgo**, no un predictor operacional. La honestidad sobre el alcance es parte de la calidad profesional del trabajo."