# Guion de defensa oral (10 minutos) — Accident_NN

> **Tesis defendible:** construimos una MLP profesional y auditable para priorización posterior a la notificación de siniestros ya fatales. La red aprende señal real, pero no supera universalmente a la logística y no debe presentarse como un predictor preventivo.

## Ruta de 10 minutos

| Tiempo | Bloque | Mensaje clave |
|---|---|---|
| 0:00-1:00 | Problema | Clasificación condicional: 1 fallecido vs. 2+ fallecidos |
| 1:00-2:15 | Datos | ONSV 2021-2025, 9,104 registros, 10.19% multifatales |
| 2:15-4:00 | Anti-fuga | Corte cronológico y 162 features disponibles al registrar el evento |
| 4:00-5:30 | Red | MLP_64_32 64-32, regularización y búsqueda 3×3 |
| 5:30-7:30 | Resultados | PR-AUC 0.2249, ROC-AUC 0.7482; logística gana F1 |
| 7:30-8:30 | Calibración + SHAP | Platt OOF; Gradient SHAP solo con train/validación |
| 8:30-10:00 | Interfaz + cierre | Cinco secciones enlazables, formulario validado y bundle verificado |

## 1. Problema — 1 minuto

> “La base ONSV contiene únicamente siniestros con al menos un fallecido. Por eso nuestra clase negativa no es ‘no fatal’: es un siniestro con exactamente un fallecido. La clase positiva es multifatal, con dos o más.”
>
> “El modelo responde una pregunta específica: **después de que un siniestro fatal fue notificado, ¿qué prioridad de revisión tiene por su probabilidad de multifatalidad?** No predice accidentes antes de que ocurran, no estima causalidad y no atribuye responsabilidad.”

## 2. Datos y target — 1 minuto 15 segundos

> “Usamos la publicación oficial preliminar ONSV 2021-2025: 9,106 filas crudas y 9,104 válidas. Hay 928 multifatales, el 10.19%. La prevalencia es importante porque un PR-AUC aleatorio sería aproximadamente 0.10.”
>
> “Excluimos `FALLECIDOS`, `LESIONADOS`, `VEHICULOS_DANADOS`, causas de investigación y señalización con faltantes dependientes del periodo. Usarlas habría producido fuga directa o información no disponible en el momento definido.”

## 3. Protocolo anti-fuga — 1 minuto 45 segundos

> “El diseño metodológico usa el orden real de los periodos:
>
> - 2021-2022: entrenamiento, 4,872 registros;
> - 2023: validación y calibración, 2,000;
> - 2024-2025: referencia histórica, 2,232.”
>
> “Escalador, encoders y frecuencias de vía se ajustan solo con entrenamiento. Arquitectura, semilla, calibrador y umbrales se eligen solo con 2023. La referencia 2024-2025 ya fue observada: no la presentamos como un periodo nuevo ni volvemos a ajustar sobre ella.”
>
> “El contrato final tiene 162 variables: ciclos de mes, día y hora; geografía; vía; escena inicial; indicadores de faltantes e interacciones predeclaradas como noche×rural, lluvia×curva, tipo de vía×zona y red vial×clase.”

## 4. Red neuronal — 1 minuto 30 segundos

> “Comparamos tres configuraciones y tres semillas. La regla de selección fue mediana de PR-AUC en validación, luego F1 mediana y menor dispersión. Ganó `MLP_64_32`: 64 y 32 neuronas ReLU, dropout 0.35, L2 de 0.0003, Adam con learning rate 0.0005, batch 64 y semilla 314.”
>
> “Usamos binary cross-entropy con pesos de clase y early stopping por PR-AUC. La robustez por semillas se muestra completa; no elegimos una corrida aislada porque salió bien.”
>
> “La regresión logística y el Random Forest recibieron las mismas 162 features, las mismas particiones y la misma política de umbral. Así evitamos favorecer a la red.”

## 5. Resultados — 2 minutos

> “En la referencia 2024-2025 hay 222 multifatales de 2,232 casos. La MLP cruda alcanza PR-AUC 0.2249 y ROC-AUC 0.7482. Es más del doble de la prevalencia en PR-AUC, así que existe señal, pero la separación sigue siendo moderada.”

| Modelo / escala | F1 | Precisión | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP cruda, umbral 0.65 | 0.3030 | 0.2346 | 0.4279 | **0.2249** | **0.7482** |
| MLP Platt, umbral 0.20 | 0.2958 | 0.2359 | 0.3964 | **0.2249** | **0.7482** |
| Logística balanceada | **0.3183** | 0.2161 | **0.6036** | 0.2054 | 0.7395 |
| Random Forest | 0.2545 | **0.2569** | 0.2523 | 0.2179 | 0.7439 |

> “Lectura honesta: la MLP tiene mejores métricas de ranking nominales en esta muestra, pero la logística mantiene mayor F1 y recall. No hicimos una prueba pareada que demuestre superioridad estadística. Por lo tanto, **no afirmamos que la red sea universalmente mejor**.”
>
> “Con la decisión calibrada tenemos 88 verdaderos positivos, 134 falsos negativos y 285 falsos positivos. Identificamos 39.6% de los multifatales y solo 23.6% de las alertas son verdaderas. Esto impide cualquier automatización operativa.”

## 6. Calibración e interpretabilidad — 1 minuto

> “Comparamos Platt e isotónica con predicciones out-of-fold de cinco particiones solo en validación 2023. Platt obtuvo menor Brier OOF. El umbral calibrado es 0.20; el score crudo conserva 0.65. Son escalas distintas y nunca se intercambian.”
>
> “En 2024-2025, Platt reduce Brier de 0.1817 a 0.0831 y ECE de 0.2623 a 0.0085. PR-AUC y ROC-AUC no cambian: calibrar corrige la escala, no el ranking. Esa distinción es fundamental.”
>
> “Gradient SHAP usa 256 registros de fondo 2021-2022 y explica 512 casos de validación 2023, con 200 muestras y semilla fija. Las 162 features se agrupan en 17 grupos. Los primeros son clase, red×clase, tipo×zona, zona y red. Son asociaciones globales, no causas ni explicaciones individuales.”

## 7. Interfaz y cierre — 1 minuto 30 segundos

> “La interfaz tiene cinco secciones enlazables: Panorama, Estimar, Explorar datos, Patrones regionales y Evidencia del modelo. Consume `models/final/` en modo read-only y verifica hashes antes de inferir.”
>
> “El formulario comienza vacío y ofrece un caso real de demostración mediante una acción explícita. Restringe la fecha a 2021-2025 y valida que el par de coordenadas esté dentro del Perú y corresponda al departamento. Los opcionales parten de NO INFORMADO y los códigos viales nuevos se identifican claramente.”
>
> “La decisión visible usa solo la probabilidad Platt; la sigmoide cruda aparece separada para auditoría. Los gráficos Plotly incluyen tablas descargables, las tasas tienen intervalos de Wilson y los subgrupos o departamentos con menos de 30 casos se enmascaran.”
>
> “AppTest recorre las cinco URLs, verifica los campos vacíos, la carga demo, un envío válido y una revalidación geográfica inválida que debe borrar el resultado previo. El gate de entrega agrega hashes, paridad de 2,232 inferencias, notebooks, PDF y estado de Git.”
>
> “Conclusión: la fortaleza del proyecto no es inflar una métrica. Es un pipeline neuronal trazable, cronológico, calibrado y honesto. El siguiente salto requiere variables reales de ocupantes, velocidad y vehículo o un periodo futuro no observado; no más ajustes sobre 2024-2025.”

---

## Preguntas probables del jurado

### 1. ¿Por qué el modelo no predice si el accidente será fatal?

Porque la fuente solo contiene accidentes ya fatales. Mezclar no fatales de otra institución haría que el modelo aprendiera diferencias de fuente. Nuestra pregunta condicional mantiene un universo coherente.

### 2. ¿La clase del siniestro no es fuga?

No para priorización posterior a la notificación: la clase puede registrarse al caracterizar el evento. Sí sería inválida para prevención antes del accidente; por eso ese uso está explícitamente prohibido.

### 3. ¿Por qué una red si la logística tiene mejor F1?

La consigna exige una red neuronal correctamente construida. La MLP muestra ranking nominalmente mejor, pero no domina todos los umbrales. Reportar la logística como competidor fuerte demuestra rigor; ocultarla sería metodológicamente indefendible.

### 4. ¿Por qué no siguieron aumentando capas?

Porque más capacidad no crea información. Con 4,872 registros de entrenamiento y 10% de positivos, el riesgo de sobreajuste crece. La búsqueda cerrada y multisemilla mostró que 64-32 regularizada era el mejor equilibrio en validación.

### 5. ¿Por qué existen dos umbrales?

Porque pertenecen a escalas distintas: 0.65 al score sigmoide crudo y 0.20 a la probabilidad Platt. La app decide solo con 0.20; 0.65 queda como diagnóstico técnico.

### 6. ¿La calibración mejoró el modelo?

Mejoró la calidad probabilística, no la discriminación. Brier y ECE bajaron; PR-AUC y ROC-AUC quedaron iguales. Si alguien afirma que la calibración hizo que el ranking fuera mejor, está confundiendo conceptos.

### 7. ¿Por qué 2024-2025 no es un test nuevo?

Porque sus resultados ya fueron observados durante el cierre experimental. Lo correcto es conservarlos como referencia histórica congelada y esperar un periodo futuro realmente no visto.

### 8. ¿SHAP prueba causalidad?

No. Explica contribuciones del modelo respecto de un fondo. El signo medio puede ocultar categorías opuestas y no dice qué ocurriría al intervenir una variable.

### 9. ¿Cuál es el punto más flojo?

La información disponible: faltan ocupantes, velocidad, vehículo, cinturón/casco y exposición. Esto explica los 134 falsos negativos y 285 falsos positivos mejor que una supuesta falta de profundidad de la red.

### 10. ¿Qué harían después?

Primero, evaluar el bundle congelado en un periodo futuro no observado. Segundo, incorporar variables reales y enlazables disponibles al momento de uso. Solo después tendría sentido reabrir arquitectura o umbrales.

## Checklist antes de exponer

- Abrir la app y las cinco secciones antes de empezar.
- Mostrar primero la tabla comparativa y la figura de evidencia, no una predicción aislada.
- Decir “referencia histórica 2024-2025”, no “periodo no observado”.
- Decir “probabilidad Platt” para la escala calibrada y “score sigmoide” para la cruda.
- No afirmar causalidad, prevención ni superioridad estadística.
- Tener abierto `report/main.pdf` en las secciones de resultados y discusión.
