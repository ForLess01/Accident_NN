# Protocolo del modelo canónico canonical-2.0.0

## Alcance
La salida estima retrospectivamente multifatalidad (2 o más fallecidos) condicionada a un siniestro ya fatal registrado. La extracción no aporta timestamps por variable para afirmar disponibilidad al instante de notificación y no demuestra causalidad.

## Particiones y selección
- Entrenamiento de la MLP congelada: 2021--2022.
- Selección de configuración completa, semilla y umbral crudo: validación 2023.
- Calibración definitiva: comparación Platt/isotónica con 5 folds OOF estratificados exclusivamente en 2023; selección por menor Brier OOF. Método seleccionado: platt.
- Umbral calibrado: máximo F1 sobre las probabilidades OOF 2023 del método seleccionado, con desempates predeclarados.
- Referencia 2024--2025: sus etiquetas ya fueron observadas. Se conserva como referencia histórica y NO puede usarse para nuevos ajustes.

## Artefacto congelado
La arquitectura y los pesos no se reentrenan ni se vuelven a buscar al materializar `models/final/`. Los umbrales crudo (0.80) y calibrado (0.30) pertenecen a escalas distintas y nunca deben intercambiarse.

## Lectura honesta
En la referencia 2024--2025 la MLP cruda obtiene PR-AUC 0.4416, ROC-AUC 0.8841 y F1 0.4957. Los liderazgos nominales derivados de la tabla canónica son: PR-AUC, Random Forest (0.4704); ROC-AUC, Random Forest (0.8937); F1, MLP cruda (0.4957). Esto no prueba superioridad universal ni estadística de la red.
