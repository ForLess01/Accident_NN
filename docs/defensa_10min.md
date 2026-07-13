# Guion de defensa oral (10 minutos) — Accident_NN

> **Tesis defendible:** construimos una MLP auditable para priorización posterior a la notificación de siniestros ya fatales. La versión 1 diagnosticó un techo informacional; la versión 2 lo demostró incorporando las bases oficiales de vehículos y personas involucradas: la discriminación casi se duplicó (PR-AUC 0.22 → 0.44) y la red pasó a superar con significación estadística al modelo lineal en ranking.

## Ruta de 10 minutos

| Tiempo | Bloque | Mensaje clave |
|---|---|---|
| 0:00-1:00 | Problema | Clasificación condicional: 1 fallecido vs. 2+ fallecidos |
| 1:00-2:15 | Datos | ONSV 2021-2025 + bases companion enlazadas al 100% por código |
| 2:15-4:00 | Anti-fuga | Corte cronológico; 175 features de escena; desenlaces por persona VETADOS |
| 4:00-5:30 | Historia v1→v2 | Diagnóstico del techo → variable faltante → salto demostrado |
| 5:30-7:30 | Resultados | PR-AUC 0.4416, ROC-AUC 0.8841, F1 0.5058; significativa vs logística en ROC |
| 7:30-8:30 | Calibración + SHAP | Platt OOF (ECE 0.017); personas involucradas = 43.7% de la importancia |
| 8:30-10:00 | Interfaz + cierre | Mapa OSM, formulario de escena, demostraciones vivas, bundle con hashes |

## 1. Problema — 1 minuto

> “La base ONSV contiene únicamente siniestros con al menos un fallecido. Por eso nuestra clase negativa no es ‘no fatal’: es un siniestro con exactamente un fallecido. La clase positiva es multifatal, con dos o más.”
>
> “El modelo responde una pregunta específica: **después de que un siniestro fatal fue notificado, ¿qué prioridad de revisión tiene por su probabilidad de multifatalidad?** No predice accidentes antes de que ocurran ni atribuye causalidad.”

## 2. Datos — 1 minuto 15 segundos

> “Usamos tres bases oficiales del ONSV 2021-2025, enlazables por código de siniestro con **cobertura del 100%**: SINIESTROS (9,104 registros útiles, 10.19% multifatales), VEHICULOS (12,667 filas, una por vehículo) y PERSONAS (25,412 filas, hasta 74 involucrados en un evento).”
>
> “De las tablas companion usamos solo hechos de escena: cuántos vehículos y de qué tipo, cuántas personas, pasajeros, peatones, fugados y edad media. Quedan **vetadas por fuga** la gravedad por persona, los lugares de atención y defunción y los dosajes etílicos: describen el desenlace o llegan con la investigación.”

## 3. Protocolo anti-fuga — 1 minuto 45 segundos

> “El diseño usa el orden real del tiempo: 2021-2022 entrena (4,872), 2023 selecciona y calibra (2,000), 2024-2025 es referencia (2,232). Escalador, encoders y frecuencias se ajustan solo con entrenamiento; arquitectura, semilla, calibrador y umbrales se eligen solo con 2023.”
>
> “El contrato v2 tiene **175 variables**. ¿Es fuga el conteo de personas involucradas? No: es la variable de exposición, un hecho de la escena con el mismo estatus que la clase del siniestro. Un evento con un solo involucrado no puede ser multifatal — eso es estructura del problema, no trampa. Lo que sería trampa es usar la gravedad por persona, y está vetada y verificada por tests.”

## 4. La historia v1 → v2 — 1 minuto 30 segundos

> “La versión 1, solo con el registro de siniestros, alcanzó PR-AUC 0.2249 y empató estadísticamente con la regresión logística. En vez de agregar capas, diagnosticamos: **el techo era informacional** — faltaba la mecánica del evento.”
>
> “La versión 2 incorporó esa información con las mismas particiones y las mismas reglas predeclaradas. La búsqueda cerrada (3 arquitecturas × 3 semillas, selección por mediana de PR-AUC) eligió `MLP_32_16`, semilla 314. Y declaramos con transparencia: la evaluación v2 es la **segunda consulta** a la referencia 2024-2025; el diseño completo se congeló antes usando solo 2021-2023.”

## 5. Resultados — 2 minutos

> “En la referencia: **PR-AUC 0.4416** [IC 0.379-0.512] con prevalencia 0.0995 —4.4 veces el azar—, **ROC-AUC 0.8841** [0.861-0.906], **F1 0.5058**, precisión 0.445, recall 0.586. Matriz calibrada: 130 aciertos multifatales, 92 escapes, 162 falsas alertas: casi 1 de cada 2 alertas es verdadera, contra 1 de cada 10 por azar.”

| Modelo | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| **MLP Platt (t=0.30)** | **0.5058** | 0.4416 | 0.8841 |
| Logística balanceada | 0.4896 | 0.4334 | 0.8580 |
| Random Forest | 0.4943 | **0.4704** | **0.8937** |

> “Bootstrap pareado de 2,000 remuestreos: contra la logística, la red gana ROC-AUC con **significación estadística** (Δ+0.026, IC [+0.012, +0.042]) — la no linealidad ahora aporta de verdad. Contra el Random Forest, empate estadístico con ventaja nominal del bosque en ranking. Reportamos ambas cosas: rigor no es ganar siempre, es medir bien.”
>
> “Y la comparación central del proyecto: v1 0.2249 → v2 0.4416 en PR-AUC sobre la MISMA referencia. La hipótesis del techo informacional quedó demostrada empíricamente.”

## 6. Calibración e interpretabilidad — 1 minuto

> “Platt se eligió por Brier OOF de 5 particiones solo en 2023 (0.0652). En la referencia, Brier 0.0683 y **ECE 0.0169**: cuando el modelo dice 30%, ocurre ≈30%. La calibración corrige la escala, no el ranking: PR-AUC y ROC-AUC no cambian.”
>
> “Gradient SHAP (fondo 2021-2022, explicados 2023, sin tocar la referencia): el grupo **personas involucradas concentra el 43.7%** de la importancia, seguido de vehículos involucrados y clase. El modelo aprendió exactamente la variable que motivó la v2. Son asociaciones globales, no causas.”

## 7. Interfaz y cierre — 1 minuto 30 segundos

> “La interfaz consume el bundle canónico `canonical-2.0.0` con hashes verificados, en solo lectura. El formulario pide los hechos de la escena —vehículos, personas, tipos— con validación de coherencia, y la ubicación se elige con un clic en un mapa OpenStreetMap acotado al Perú que deduce el departamento por point-in-polygon.”
>
> “El Panorama demuestra el modelo contra la realidad: la serie mensual predicho vs. observado, la multifatalidad real por quintil de score y la comparación por categoría, todo sobre 2024-2025 con intervalos de Wilson.”
>
> “Conclusión: el proyecto entrega método y resultado. Diagnosticamos un límite con honestidad, lo atacamos con datos oficiales enlazables y demostramos el salto con el mismo protocolo congelado. El siguiente paso válido es un periodo futuro no observado; el siguiente techo, velocidad y protección de ocupantes.”

---

## Preguntas probables del jurado

### 1. ¿El número de personas involucradas no es fuga del resultado?

No. Es un hecho de la escena disponible al caracterizar la notificación (incluye ilesos) y constituye la exposición del evento: define el techo de víctimas posibles, no el desenlace. La fuga sería usar la gravedad por persona, que está vetada y las pruebas lo verifican. Declaramos como limitación que el registro de ilesos podría ser más completo en siniestros graves.

### 2. Evaluaron dos veces sobre 2024-2025. ¿Eso no invalida la v2?

Lo divulgamos explícitamente en el informe. Ninguna decisión de la v2 usó esas etiquetas: contrato, grilla, semillas, calibrador y umbrales se congelaron con 2021-2023. El riesgo de multiplicidad con UNA evaluación previa es mínimo y el salto (ranking casi duplicado) excede largamente lo atribuible a ese efecto. La validación definitiva exige un periodo futuro; lo decimos nosotros antes de que lo pregunten.

### 3. ¿Por qué la red y no el Random Forest, que tiene mejor PR-AUC?

Empate estadístico (ΔPR-AUC −0.029, IC [−0.073, +0.014]): no hay evidencia de que el bosque sea mejor, ni de lo contrario. La red sí supera significativamente al modelo lineal en ROC-AUC, cumple la consigna del curso y aporta la infraestructura de calibración y explicabilidad del sistema. Reportar al bosque como competidor de primer nivel es parte del rigor.

### 4. ¿Por qué cambió la arquitectura entre versiones?

Porque la regla predeclarada decide, no la preferencia: con 175 features la mediana de PR-AUC entre semillas favoreció a `MLP_32_16` (0.4806) sobre `MLP_64_32` (0.4682). Mantener la arquitectura v1 por apego habría sido selección arbitraria.

### 5. ¿Qué significa el umbral calibrado 0.30?

Se eligió sobre probabilidades OOF de 2023 por máximo F1 con desempates predeclarados. La decisión pública usa SOLO la escala Platt; el score crudo (umbral 0.80) queda en un panel de auditoría. Las escalas nunca se mezclan.

### 6. ¿Por qué no predice si un accidente será fatal?

Porque la fuente contiene solo accidentes ya fatales; mezclar no-fatales de otra institución enseñaría al modelo a distinguir fuentes, no severidad. La pregunta condicional mantiene un universo coherente.

### 7. ¿Qué pasa con un código de vía que el modelo nunca vio?

Nada se rompe: mapea a frecuencia cero por contrato, la interfaz lo advierte y está cubierto por tests.

### 8. ¿Por qué no usaron SMOTE?

Con 10.2% de prevalencia, pesos de clase bastan; sintetizar combinaciones categóricas y geográficas crearía casos indefendibles. Regla declarada antes de entrenar.

---

## Notas para el ensayo

- **Ensayar con cronómetro**; si se pasa de 10, recortar la demo (mostrar solo Panorama: quintiles + serie mensual).
- **Tener abiertas las cinco secciones** de la app antes de empezar.
- **Tener el PDF en la tabla v1 vs v2** (Sección de Resultados): es el gráfico mental que el jurado se lleva.
- **Cerrar con la frase**: “Diagnosticamos el límite, conseguimos la variable que faltaba y demostramos el salto con el protocolo congelado. Eso es ingeniería de aprendizaje automático, no prueba y error.”
