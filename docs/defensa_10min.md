# Defensa técnica de Accident_NN (10 minutos)

> **Tesis defendible:** construimos una MLP tabular compacta y auditable para clasificar retrospectivamente multifatalidad entre siniestros fatales registrados. La red mejora el ranking frente a una regla de conteo y a la regresión logística, pero no domina al Random Forest ni autoriza conclusiones causales u operativas.

## Ruta

| Tiempo | Bloque | Mensaje verificable |
|---|---|---|
| 0:00-1:00 | Pregunta | 1 fallecido vs. 2+ fallecidos, dentro de siniestros ya fatales |
| 1:00-2:15 | Datos | ONSV 2021-2025, 9.104 registros; 26 campos crudos → 175 features |
| 2:15-4:15 | Red | 175→32→16→1; ReLU, L2, dropout, BCE ponderada, Adam |
| 4:15-5:30 | Protocolo | 2021-22 entrena; 2023 selecciona/calibra; 2024-25 referencia histórica |
| 5:30-7:30 | Resultados | PR-AUC 0.4416; ROC-AUC 0.8841; F1 0.5058; IC y baselines |
| 7:30-8:45 | Justificación | Ablación, una-vs-varias redes, baseline `n_personas` |
| 8:45-10:00 | Interfaz y límites | Flujo visible, segunda consulta, disponibilidad temporal y próximos datos |

## Guion

### 1. Pregunta y datos

“La clase positiva es `FALLECIDOS >= 2`; la negativa es exactamente un fallecido. No estimamos fatalidad sobre todos los accidentes. Usamos 9.104 registros ONSV y dos tablas complementarias de vehículos y personas. El formulario tiene 26 campos crudos y el preprocesamiento produce 175 features en orden fijo.”

“La extracción no contiene timestamps por variable. Por eso no afirmamos que todos los campos companion estuvieran disponibles al instante de notificación: el alcance defendible es retrospectivo o post-registro.”

### 2. Arquitectura y aprendizaje

“La única red canónica es 175→Dense(32, ReLU)→Dropout(0.25)→Dense(16, ReLU)→Dropout(0.25)→Dense(1, sigmoide). Tiene dos capas ocultas, tres capas densas entrenables y 6.177 parámetros.”

“El forward pass transforma las 175 entradas con ReLU y produce un score sigmoide. Backpropagation propaga el gradiente de la BCE ponderada más L2 hacia atrás; Adam actualiza los pesos. Los pesos de clase atienden el desbalance; L2 y dropout controlan capacidad; early stopping y ReduceLROnPlateau controlan la trayectoria de entrenamiento.”

“Platt es externo: `p = sigmoid(a * logit(s) + b)`. No es otra red, una capa de la MLP ni stacking.”

### 3. Selección sin fuga

“2021-2022 ajusta preprocesamiento y pesos; 2023 selecciona la configuración completa, semilla, calibrador y umbrales. Son tres configuraciones por tres semillas, nueve corridas. Digo configuración porque cambian simultáneamente capas, dropout, L2 y learning rate; la búsqueda no aísla arquitectura.”

“2024-2025 es referencia histórica y la v2 es la segunda consulta declarada. Congelar el diseño reduce tuning directo, pero el riesgo de multiplicidad no fue cuantificado; solo un periodo futuro o externo confirma.”

### 4. Resultados

| Modelo / escala | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| **MLP Platt, t=0.30** | **0.5058** | 0.4416 | 0.8841 |
| Logística balanceada | 0.4896 | 0.4334 | 0.8580 |
| Random Forest | 0.4943 | **0.4704** | **0.8937** |

“La MLP tiene PR-AUC 0.4416 [0.3785, 0.5124], ROC-AUC 0.8841 [0.8613, 0.9055] y F1 0.5058. Contra logística, ΔROC-AUC es +0.0261 [0.0122, 0.0417]. Contra Random Forest no se detecta diferencia significativa; eso no significa equivalencia.”

“La matriz calibrada es TN 1.848, FP 162, FN 92 y TP 130. Los 92 FN son 41.4 % de 222: aproximadamente dos de cada cinco.”

### 5. ¿Son necesarias estas técnicas y una sola red?

“La ablación usa solo 2021-2023. L2+dropout logra PR-AUC mediana 0.4806 y ROC-AUC mediana 0.8911. Solo dropout llega a PR-AUC 0.4818; solo L2 a 0.4655; sin ambos a 0.4648. La evidencia apoya mantener el paquete estable, pero no demuestra que cada regularizador sea indispensable por separado.”

“El ensemble de tres semillas sube nominalmente PR-AUC de 0.4806 a 0.4976, pero los IC pareados de PR-AUC, ROC-AUC y F1 incluyen cero. La multirrama 162+13 obtiene 0.4622. Por parsimonia mantenemos una sola red; el ensemble exige nuevo holdout y calibración.”

“La regla simple `n_personas >= 4`, elegida solo en 2023, obtiene PR-AUC 0.3941 y ROC-AUC 0.8540 en la referencia. La MLP mejora +0.0475 [0.0167, 0.0833] y +0.0301 [0.0120, 0.0491]. En F1 no se detecta diferencia. La red aporta ranking más allá del conteo, no una victoria en toda métrica.”

### 6. Cierre

“La interfaz muestra el flujo completo 26→175→32→16→Platt→clase, cinco escenarios controlados, curvas, calibración, matriz, bootstrap, ablaciones y SHAP global. Comparar escenarios muestra respuesta del modelo, no causalidad.”

“El aporte profesional no es agregar capas: es demostrar qué se seleccionó, qué no se justificó, dónde gana, dónde no, y qué evidencia falta.”

---

## Preguntas difíciles

### ¿Por qué una red si Random Forest lidera nominalmente?

Porque la consigna evalúa una red neuronal y la MLP es competitiva, calibrable y auditable. Random Forest lidera PR-AUC y ROC-AUC nominales; los IC pareados no detectan diferencia. No ocultamos ese resultado ni afirmamos equivalencia.

### ¿Por qué una sola red y no dos?

El ensemble mejora nominalmente, pero todos los IC pareados incluyen cero. Una segunda red agrega mantenimiento y exige recalibración sin evidencia concluyente. La multirrama tampoco mejora. Se conserva una única MLP parsimoniosa.

### ¿La MLP solo cuenta personas?

No. La regla `n_personas >= 4` tiene F1 similar, pero la MLP la supera significativamente en PR-AUC y ROC-AUC en la comparación post-hoc de referencia. Eso muestra mejor ordenamiento con el conjunto de variables.

### ¿Por qué 32 y 16 neuronas?

No por intuición aislada: la configuración 32-16 ganó la regla predeclarada de mediana PR-AUC entre tres semillas. Es compacta para 4.872 registros y tiene 6.177 parámetros. La búsqueda compara paquetes completos, no solo capas.

### ¿L1, batch normalization, SMOTE o stacking mejorarían?

No hay evidencia actual que lo justifique. L1 puede forzar esparsidad redundante con el contrato; batch normalization no es necesaria en esta MLP tabular escalada; SMOTE sintetizaría combinaciones geográficas/categóricas; stacking cambia el sistema y necesita holdout. Agregarlos sin validación nueva sería complejidad, no rigor.

### ¿Hay sobreajuste train/validation?

Se controla con red compacta, L2, dropout, early stopping, tres semillas y selección agregada. La estabilidad entre semillas y la referencia posterior son compatibles con generalización, pero no eliminan incertidumbre ni sustituyen validación externa.

### ¿La segunda consulta invalida los resultados?

No los invalida, pero reduce su fuerza confirmatoria. Ninguna decisión v2 usó 2024-2025 directamente; aun así, el riesgo por consultas repetidas no está cuantificado. Por eso se rotula referencia histórica y se exige un periodo futuro para confirmación.

### ¿Los campos companion estaban disponibles al notificar?

No se puede demostrar con esta extracción porque faltan timestamps por variable. Son datos del registro consolidado; la interfaz y el informe los tratan retrospectivamente. Una operación en tiempo real exige metadatos de disponibilidad.

### ¿Qué significa calibración?

Platt mejora la concordancia agregada entre probabilidad media y frecuencia observada; no garantiza que un caso con 30 % “ocurra 30 %”. Brier y ECE evalúan grupos, mientras PR-AUC y ROC-AUC miden ranking.

### ¿No significativo significa equivalentes?

No. Significa que con esta muestra y procedimiento no se detectó una diferencia. Demostrar equivalencia requiere márgenes y una prueba de equivalencia preespecificados.

### ¿SHAP explica causas?

No. Gradient SHAP resume asociaciones del score global respecto de un fondo de entrenamiento. El signo medio no es una intervención, monotonicidad ni explicación causal individual.

## Antes de presentar

- Abrir `?section=panorama`, `?section=estimar` y `?section=evidencia`.
- Mostrar primero la arquitectura y luego el forest plot; no empezar por una métrica aislada.
- Decir “no se detectó diferencia significativa”, nunca “empate” o “equivalencia”.
- Reconocer que 2025 es parcial y que la disponibilidad temporal no fue demostrada.
