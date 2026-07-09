# Auditoría del modelo frente al Plan.md

## Veredicto

El modelo es metodológicamente correcto para la definición reformulada del proyecto: clasificación binaria de alta letalidad en siniestros fatales (`target_multifatal = 1` si `FALLECIDOS >= 2`), salida sigmoide, evaluación con métricas para desbalance y comparación contra baselines. No se detectaron features derivadas de fallecidos, lesionados, vehículos dañados, causas post-investigación ni señalización excluida, por lo que no hay evidencia de fuga directa del target en `feature_list.json`.

## Evidencia

- Features finales: 116.
- Features con posible fuga: 0.
- F1-multifatal MLP en validación: 0.2929.
- Mejor F1-multifatal baseline en validación: 0.2910.
- F1-multifatal MLP en test: 0.2746.
- Mejor F1-multifatal baseline en test: 0.3081 (LogisticRegression_balanced).
- Recall-multifatal MLP en test: 0.8849.
- PR-AUC MLP en test: 0.2038.
- ROC-AUC MLP en test: 0.7474.
- Evaluaciones del test registradas: 1.
- Calibrador post-hoc seleccionado: isotonic.
- Brier crudo/calibrado en test: 0.2139 / 0.0872.
- ECE crudo/calibrado en test: 0.3154 / 0.0161.

## ¿Necesita mejorarse?

No necesita corregirse para cumplir el plan: usa el split correcto, mantiene el test como evaluación final única, evita leakage y privilegia el recall del evento de mayor costo. Sí debe presentarse con honestidad: la regresión logística gana nominalmente en F1 de test, mientras que el MLP gana en recall multifatal (0.8849) y mantiene ROC-AUC comparable. Ese empate con un baseline clásico es esperable en datos tabulares medianos y no debe maquillarse.

La mejora aplicada no cambia la red ni maquilla el F1: calibra la lectura del riesgo con validation. El umbral de máxima sensibilidad en validación fue 0.10, con recall 1.0000 pero F1 0.2050. Cambiar ahora el umbral después de haber visto el test no sería metodológicamente limpio; se debe reportar esta tensión en Discusión y proponer como trabajo futuro una política de umbral definida por costo operacional antes de evaluar un nuevo holdout.
