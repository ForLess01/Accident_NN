# Protocolo del modelo canónico canonical-1.0.0

## Alcance
La salida estima multifatalidad (2 o más fallecidos) condicionada a un siniestro ya fatal. Su uso defendible es priorización posterior a la notificación; no predice que un siniestro cualquiera vaya a ser fatal ni demuestra causalidad.

## Particiones y selección
- Entrenamiento de la MLP congelada: 2021--2022.
- Selección de arquitectura, semilla y umbral crudo: validación 2023.
- Calibración definitiva: comparación Platt/isotónica con 5 folds OOF estratificados exclusivamente en 2023; selección por menor Brier OOF. Método seleccionado: platt.
- Umbral calibrado: máximo F1 sobre las probabilidades OOF 2023 del método seleccionado, con desempates predeclarados.
- Referencia 2024--2025: sus etiquetas ya fueron observadas. Se conserva como referencia histórica y NO puede usarse para nuevos ajustes.

## Artefacto congelado
La arquitectura y los pesos no se reentrenan ni se vuelven a buscar al materializar `models/final/`. Los umbrales crudo (0.65) y calibrado (0.20) pertenecen a escalas distintas y nunca deben intercambiarse.

## Lectura honesta
En la referencia 2024--2025 la MLP tiene PR-AUC 0.2249 y ROC-AUC 0.7482. Sus métricas de ranking son nominalmente mayores que las de los baselines declarados, pero la regresión logística conserva mayor F1; esto no prueba superioridad universal ni estadística de la red.
