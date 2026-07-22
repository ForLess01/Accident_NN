# Defensa de 10 minutos - modelo definitivo

**Autores:** Rendo Alfonte Tarqui; Yimmy Yeyson Cuyo Zamata

| Tiempo | Punto |
|---|---|
| 0:00-1:00 | Objetivo: multifatalidad condicionada a siniestros ya fatales, no predicción de accidentes |
| 1:00-2:00 | Procedencia ONSV, hashes y prueba de fuga PERSONAS |
| 2:00-4:00 | 21 entradas -> 169 features -> 64 -> 32 -> 1; 12.993 parámetros |
| 4:00-5:30 | Fit 2021, selección 2022, reajuste 2021-22, calibración/umbral 2023 |
| 5:30-7:00 | Referencia 2024-25: PR-AUC 0,325; ROC-AUC 0,792; F1 0,399 |
| 7:00-8:00 | Una red, regularización real y folds rolling diagnósticos |
| 8:00-9:00 | Interfaz: cinco roles pedagógicos y escalas separadas |
| 9:00-10:00 | Límites: referencia ya consultada, sin cohorte externa ni intervalo individual |

## Explicación exacta de la red

“Tenemos una sola MLP. La entrada procesada tiene 169 variables. La primera capa oculta tiene 64 neuronas ReLU; sigue dropout 0,35. La segunda tiene 32 neuronas ReLU y otro dropout 0,35. La salida es una neurona sigmoide. Las tres capas Dense suman 12.993 parámetros entrenables.”

“Usamos BCE con pesos de clase, Adam con tasa inicial 0,0005, L2 de 0,0003, dropout, early stopping y reducción de learning rate en la selección. No usamos L1, SMOTE, stacking ni batch normalization porque no había evidencia que justificara agregar complejidad.”

## Pregunta difícil: ¿por qué eliminar PERSONAS?

“Porque no era un simple riesgo teórico. La auditoría por código mostró que el conteo de filas PERSONAS cubre necesariamente a los fallecidos y que contar `GRAVEDAD=FALLECIDO` reproduce `FALLECIDOS` en 9.104 de 9.104 casos. Como el target es `FALLECIDOS >= 2`, cualquier agregado de esa tabla está estructuralmente contaminado. La corrección profesional fue excluir la fuente completa de las entradas.”

## Pregunta difícil: ¿cómo seleccionaron sin usar el futuro?

“La arquitectura y semilla se eligen con fit 2021 y selección 2022. El modelo final se reajusta una vez en 2021-2022 durante 14 épocas fijadas por la selección interna. En la partición 2023 se validan Platt y el umbral, sin volver a buscar arquitectura. Además, ejecutamos dos diagnósticos rolling donde fit, selección, calibración y outer están cronológicamente disjuntos dentro de cada fold.”

## Pregunta difícil: ¿los resultados son excelentes?

“No. Son honestos y moderados. En 2024-2025, ya consultado, la MLP tiene PR-AUC 0,325 frente a prevalencia 0,099, ROC-AUC 0,792 y F1 calibrado 0,399. La calibración mejora Brier de 0,164 a 0,078 y ECE de 0,234 a 0,013, pero no mejora ranking. No presentamos esa referencia como test confirmatorio.”

## Pregunta difícil: ¿la MLP supera a Random Forest?

“No podemos afirmar superioridad general. En el bootstrap pareado, las diferencias de PR-AUC y ROC-AUC frente a Random Forest incluyen cero. F1 es la excepción nominal: MLP menos Random Forest es +0,0252, con IC 95 % [0,00166; 0,04790], favorable a la MLP. Son seis comparaciones con intervalos nominales al 95 %, sin ajuste por multiplicidad ni cobertura familiar simultánea. Es evidencia condicional a las predicciones congeladas, no una validación externa ni incertidumbre de reentrenamiento.”

## Pregunta difícil: ¿por qué una red y no dos?

“El ensemble de tres semillas no mostró una ventaja concluyente en los IC pareados de la auditoría. Una segunda red agregaría selección y calibración sin evidencia suficiente. Por parsimonia se conserva una sola MLP.”

## Incertidumbre

“El bootstrap es condicional a predicciones congeladas: no reentrena. Por eso no cubre selección, dependencia temporal/espacial, consultas repetidas ni futuro. No tenemos un intervalo validado para una predicción individual; los Wilson de la interfaz son tasas agregadas.”

## Cierre

“La entrega es definitiva como artefacto académico, no confirmatoria como ciencia externa. Una confirmación real exige datos futuros o externos intactos.”
