# Guion de Defensa Oral (10 minutos) — Accident\_NN

> Proyecto: Clasificación de alta letalidad en siniestros viales fatales del Perú.
> Equipo: **Rendo** (líder técnico, macOS) y **Yimmy** (Windows).
> Curso: Redes Neuronales — Segunda Unidad.

---

## Estructura de la defensa (10 minutos en total)

| Tiempo | Bloque | Punto fuerte |
|---|---|---|
| 0:00--1:00 | Problema | Encuadre condicional: dado un siniestro fatal, ¿será multifatal? |
| 1:00--3:00 | Dataset y EDA | H8 (rural 15.1 % vs urbano 3.6 %) y H5 (despiste domina la letalidad) |
| 3:00--5:00 | Anti-fuga + features | Exclusión de lesionados/causas + 116 features pre-impacto |
| 5:00--7:00 | Red + entrenamiento | Grid cerrado, early stopping con PR-AUC, umbral 0.35 por costo asimétrico |
| 7:00--9:00 | Resultados + SHAP | ROC-AUC 0.747, recall 0.88, calibración isotónica, bootstrap |
| 9:00--10:00 | Demo GUI en vivo | 4 pestañas, gauge calibrado, caso curva+lluvia, mapa coroplético |

---

## 1. Problema (1 min)

> "La siniestralidad vial es un problema de salud pública en el Perú. Dentro de los siniestros fatales, una fracción concentra múltiples víctimas en un solo evento: buses interprovinciales, vías rurales, condiciones adversas. Nuestra pregunta: **dado que ocurre un siniestro fatal, ¿qué condiciones se asocian a que sea de alta letalidad (2+ fallecidos), y puede una red neuronal clasificarlas?**"
>
> "Elegimos **clasificación binaria multifatal / letalidad simple**, no regresión: el conteo de fallecidos dentro de los fatales está muy sesgado (89.8 % con exactamente uno, cola hasta 33), la pregunta útil es binaria, y la clasificación permite la batería completa de métricas para desbalance. El encuadre es **condicional** porque la base ONSV registra solo siniestros fatales — no existe clase 'no fatal', y mezclar fuentes con criterios distintos habría introducido sesgo de selección."

## 2. Dataset y EDA (2 min)

> "Usamos la base oficial **SINIESTROS DE TRANSITO FATALES 2021--2025 (PRELIMINAR)**, publicada por ONSV el 2026-02-27: 9\,106 registros crudos, 9\,104 útiles. La clase multifatal representa el **10.19 %**, que define el desbalance y el régimen metodológico. Es la primera fuente pública peruana con variables pre-impacto a nivel de registro: zona, red vial, clima, curvatura, perfil, superficie y coordenadas."

> "**Dos hallazgos fuertes del EDA:**
>
> - **H8:** la zona rural presenta tasa multifatal de **15.1 %** contra **3.6 %** en zona urbana — 4.2 veces más. Y el clima lluvioso 17.6 % contra 9.8 % en despejado. Estas variables no existían en la fuente Sutran de nuestra primera iteración.
> - **H5:** CHOQUE es la clase más frecuente, pero **DESPISTE** tiene la mayor tasa multifatal (17.7 %) — los despistes de buses y camiones en vías rurales concentran los eventos con múltiples víctimas."

## 3. Anti-fuga + features (2 min)

> "Tenemos **dos guardas anti-fuga**. Claridad metodológica absoluta:
>
> 1. Las columnas de **resultado** están prohibidas como entradas: `FALLECIDOS` define el target; `LESIONADOS` y `VEHICULOS_DANADOS` se cuentan después del siniestro; la `CAUSA` se determina en la investigación posterior (49 % figura 'en proceso'). El contrato de features lo verifica: **de 116 features finales, ninguna deriva de esas columnas.**
> 2. **Primero split, después todo fit.** Cualquier estadístico se calcula SOLO con train. Split estratificado 70/15/15 con `random_state=42`: 10.20 / 10.18 / 10.18 % por subconjunto."

> "Sobre features: construimos **116 características**. Las tres decisiones técnicas más importantes:
>
> - **Codificación cíclica de hora** (`sin`/`cos`): para la red, 23 y 0 deben ser vecinos.
> - **Coordenadas estandarizadas** con flag de faltante: señal espacial continua que complementa al departamento.
> - **Categorías cerradas** para clima, curvatura, superficie y clase (p. ej. 'ATROPELLO FUGA' → ATROPELLO), con `DESCONOCIDO` explícito en lugar de imputaciones inventadas."

## 4. Red + entrenamiento (2 min)

> "El modelo es un **MLP** — el estándar para datos tabulares. Tras un **grid cerrado de 6 corridas definido ANTES de entrenar**, ganó la corrida R5: una sola capa oculta de 32 neuronas, dropout 0.4, batch normalization, F1-multifatal en validación 0.2929."

> "Detalles técnicos profesionales:
> - Pérdida: **entropía cruzada binaria** con `class_weight='balanced'` (sin SMOTE, porque 10.2 % está en el rango 5--25 %).
> - Optimizador: **Adam**, LR 1e-3. Early stopping monitoreando **PR-AUC** con restore best weights.
> - **Umbral 0.35 elegido en validación** con una regla declarada antes de entrenar: máximo recall entre los umbrales que conservan F1 ≥ 90 % del mejor. Elegimos recall 0.92 sobre F1 0.30 porque el **falso negativo** (subestimar un siniestro con múltiples víctimas) es el error más caro del dominio."

## 5. Resultados + SHAP (2 min)

> "El **test se evaluó exactamente una vez**. Resultados (umbral MLP 0.35; baselines 0.5):
>
> | Modelo | F1-multifatal | Recall | ROC-AUC |
> |---|---|---|---|
> | Dummy | 0.00 | 0.00 | 0.500 |
> | Reg. Logística | **0.308** | 0.763 | **0.751** |
> | Random Forest | 0.178 | 0.137 | 0.736 |
> | **MLP** | 0.275 | **0.885** | 0.747 |
>
> Lectura honesta: la **logística gana nominalmente en F1 y ROC-AUC**, con intervalos bootstrap traslapados — empate estadístico. El MLP entrega el **mayor recall (0.885)**, que es la métrica que nuestra regla de costo privilegia. Declararlo así es la contingencia C13 y suma credibilidad."

> "**Top-5 SHAP**: `clase_atropello`, `zona_rural`, `clase_despiste`, `clase_choque`, `zona_urbana`. La red aprendió **exactamente las variables de contexto que motivaron el cambio de fuente** — coherencia total con el EDA (H5, H8)."

> "**Calibración**: la sigmoide cruda estaba mal calibrada (Brier 0.214). Ajustamos **calibración isotónica solo con validación**: Brier 0.087 en el test congelado, **mejor que el predictor de tasa base (0.092)**. El gauge de la GUI muestra probabilidades reales. Y reportamos **intervalos bootstrap al 95 %** en todas las métricas: con 139 positivos en test, venderlas como puntos exactos sería deshonesto."

## 6. Demo GUI en vivo (1 min)

> "Cuatro pestañas en Streamlit, todas consumen el mismo `preparar_entrada()` (cero lógica duplicada). Demo en vivo:
> - Pestaña 1: cargo `demo_01` (choque diurno, carretera nacional rural) → probabilidad calibrada **14 %**.
> - Cargo `demo_03` (curva + lluvia) → sube a **20 %**, la sigmoide cruda de 0.65 a 0.76. Coherente con H8.
> - Pestaña 2: EDA con los hallazgos H1--H11 y el scatter geográfico.
> - Pestaña 3: ranking departamental (LIMA por volumen, HUANCAVELICA por tasa: 28 %) + **mapa coroplético del Perú**.
> - Pestaña 4: métricas del test, calibración y top-5 SHAP."

---

## 5 preguntas más probables del jurado — guion de respuesta

### P1. ¿Por qué todos los registros son fatales? ¿Dónde está la clase negativa?
- El problema es **condicional**: dado un siniestro fatal, ¿deja 1 o 2+ fallecidos? La clase negativa es 'letalidad simple' (1 fallecido, 89.8 %).
- Es un problema reconocido en la literatura de seguridad vial (*multiple-fatality crashes*). La alternativa — mezclar fatales ONSV con no-fatales de otra fuente — introduciría **sesgo de selección**: el modelo aprendería a detectar la fuente, no la letalidad.

### P2. ¿Por qué no usaron `LESIONADOS` o la `CAUSA` como entrada?
- `LESIONADOS` y `VEHICULOS_DANADOS` se cuentan **después** del siniestro y correlacionan con la clase (media de afectados 1.4 vs 5.0): usarlos sería *data leakage* del resultado.
- La `CAUSA` se determina en la **investigación posterior** — 49 % figura 'en proceso de investigación'. Para un modelo predictivo, no se conoce en el momento del evento.

### P3. ¿Por qué la accuracy es solo 0.52?
- Consecuencia **deliberada** del umbral 0.35 orientado a recall: preferimos alarmar de más antes que perder un evento multifatal (16 falsos negativos de 139). La precision de la clase de letalidad simple es 0.97: cuando el modelo descarta, casi siempre acierta.
- La métrica correcta bajo desbalance es F1/PR-AUC/recall, no accuracy: el Dummy alcanza 0.90 de accuracy y es inútil.

### P4. La logística le gana a la red en F1. ¿Entonces para qué la red?
- Empate estadístico: los intervalos bootstrap se traslapan ampliamente. En tabulares pequeños los modelos lineales compiten de igual a igual — fenómeno conocido y declarado desde el diseño (C13).
- El requisito del curso es entrenar y evaluar **correctamente** una red neuronal, con baselines honestos. El MLP además entrega el mayor recall (0.885), la métrica privilegiada por la regla de costo.

### P5. ¿Qué significa el umbral 0.35 y por qué no 0.5?
- Regla declarada **antes** de entrenar: máximo recall-multifatal entre umbrales con F1 ≥ 90 % del mejor F1 de validación. 0.65 daba el mejor F1 (0.304) pero recall 0.53 — la mitad de los eventos multifatales pasaría desapercibida.
- Distinción clave: el **umbral** decide la etiqueta sobre la sigmoide cruda; la **calibración isotónica** corrige la probabilidad que se comunica (Brier 0.214 → 0.087).

---

## 3 preguntas adicionales de respaldo

### P6. ¿Por qué cambiaron de dataset a mitad del proyecto?
- La primera iteración (Sutran 2020--2021, mortal/no-mortal) alcanzó ROC-AUC 0.684 con una red que empataba con la logística: diagnóstico de **techo informacional** (solo 6 variables base, sin contexto).
- El cambio a ONSV agregó las variables pre-impacto que faltaban y elevó el ROC-AUC a 0.747, con SHAP dominado por las variables nuevas. La hipótesis del techo se **confirmó empíricamente** — eso es método, no improvisación.

### P7. ¿Qué pasa si la app recibe un código de carretera que nunca vio?
- Nada se rompe: `via_freq` se mapea a 0 (frecuencia calculada en train). Es el caso `demo_04_codigo_no_visto` y se prueba en el checklist de la GUI.

### P8. ¿Por qué SMOTE no se usó?
- Regla de contingencia del Plan: si la clase minoritaria está entre 5 % y 25 %, solo `class_weight`. Como tenemos 10.2 %, aplicamos C5: `class_weight='balanced'` sin SMOTE.

---

## Notas para el ensayo

- **Ensayar en vivo** con cronómetro: cada bloque debe durar el tiempo asignado. Si en la práctica se va a 11 min, recortar la demo (mostrar solo pestaña 1 con el caso curva+lluvia).
- **Tener abiertas las 4 pestañas** de la app antes de empezar, para no perder tiempo en la demo.
- **Tener el PDF del informe abierto** en la tabla de bootstrap (Sección 10) por si preguntan por la incertidumbre.
- **Mantener tono formal académico**: pasiva refleja ("se entrenó", "se observó"), sin primera persona.
- **Cerrar con una frase de alcance**: "El modelo es una **estimación académica de factores de riesgo condicionales**, no un predictor operacional. La honestidad sobre el alcance es parte de la calidad profesional del trabajo."