# Protocolo del modelo canónico canonical-3.0.0

## Alcance
La salida estima retrospectivamente multifatalidad (2 o más fallecidos) condicionada a un siniestro ya fatal registrado. Todo agregado de PERSONAS está excluido porque su cardinalidad y el conteo de fallecidos reconstruyen el objetivo. La extracción no aporta timestamps por variable para afirmar disponibilidad al instante de notificación y no demuestra causalidad.

## Cronología canónica
- Selección de arquitectura, configuración y semilla: ajuste en 2021 y comparación en 2022.
- Reajuste final de la MLP congelada: datos 2021--2022.
- Validación de calibración y umbrales: exclusivamente 2023; no se busca arquitectura en este periodo.
- Referencia histórica 2024--2025: etiquetas ya observadas; no se usa para ajustar arquitectura, calibración ni umbrales.
- Épocas del reajuste final: 14, fijadas por el protocolo de selección de 2022.
- Calibración: comparación Platt/isotónica con 5 folds OOF estratificados y criterio de menor Brier OOF. Método resultante: platt.
- Umbral crudo y calibrado: máximo F1 de validación; el calibrado usa las probabilidades OOF del método resultante. Ambos aplican los desempates predeclarados.

## Artefacto congelado
La arquitectura y los pesos no se reentrenan ni se vuelven a buscar al materializar `models/final/`. Los umbrales crudo (0.65) y calibrado (0.15) pertenecen a escalas distintas y nunca deben intercambiarse. La finalidad del artefacto de entrega no convierte 2024--2025 en una cohorte confirmatoria independiente.

## Lectura honesta
En la referencia 2024--2025 la MLP cruda obtiene PR-AUC 0.3254, ROC-AUC 0.7916 y F1 0.4030. Los liderazgos nominales derivados de la tabla canónica son: PR-AUC, MLP cruda (0.3254); ROC-AUC, MLP cruda (0.7916); F1, MLP cruda (0.4030). Frente a Random Forest, PR-AUC y ROC-AUC no muestran diferencia nominal; F1 sí favorece nominalmente a la MLP por 0.0252, IC95% [0.0017, 0.0479]. Son seis comparaciones con intervalos nominales al 95% sin ajuste por multiplicidad; no se ofrece cobertura familiar simultánea ni se prueba superioridad universal.
