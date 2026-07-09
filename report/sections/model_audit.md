# Auditoría del modelo frente al Plan.md

## Veredicto

El modelo es metodológicamente correcto para la definición del proyecto: clasificación binaria mortal/no mortal, salida probabilística sigmoide, evaluación con métricas para desbalance y comparación contra baselines. No se detectaron features con `FALLECIDOS` ni `HERIDOS`, por lo que no hay evidencia de fuga directa del target en `feature_list.json`.

## Evidencia

- Features finales: 72.
- Features con posible fuga: 0.
- F1-mortal MLP en validación: 0.3077.
- Mejor F1-mortal baseline en validación: 0.2867.
- F1-mortal MLP en test: 0.2941.
- Mejor F1-mortal baseline en test: 0.2719.
- Recall-mortal MLP en test: 0.4895.
- PR-AUC MLP en test: 0.2890.
- Evaluaciones del test registradas: 1.
- Calibrador post-hoc seleccionado: isotonic.
- Brier crudo/calibrado en test: 0.2017 / 0.0970.
- ECE crudo/calibrado en test: 0.3171 / 0.0169.

## ¿Necesita mejorarse?

No necesita corregirse para cumplir el plan: supera al Dummy y a los baselines en F1-mortal, usa el split correcto y mantiene el test como evaluación final. Sí debe presentarse como un modelo de desempeño limitado: el recall-mortal de test es 0.4895, por lo que todavía deja falsos negativos relevantes.

La mejora aplicada no cambia la red ni maquilla el F1: calibra la lectura del riesgo con validation. El umbral de máxima sensibilidad en validación fue 0.10, con recall 1.0000 pero F1 0.2104. Cambiar ahora el umbral después de haber visto el test no sería metodológicamente limpio; se debe reportar esta tensión en Discusión y proponer como trabajo futuro una política de umbral definida por costo operacional antes de evaluar un nuevo holdout.
