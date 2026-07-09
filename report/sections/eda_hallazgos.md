# Hallazgos del Bloque C — EDA

H1: La clase mortal representa 11.7% del dataset limpio (fig01). Esto confirma el desbalance y justifica evaluar con F1, recall y PR-AUC, no con accuracy aislada.
H2: El mes con más accidentes fue 2021-08 con 560 registros (fig02). La serie mensual muestra que la frecuencia no es constante, por lo que las variables temporales aportan señal al modelo.
H3: La hora con más accidentes fue las 17:00 con 507 registros (fig03). En cambio, la mayor tasa de mortalidad por hora aparece a las 01:00 con 18.8% (fig04), mostrando que volumen y severidad no son lo mismo.
H4: La comparación por día de semana (fig05) muestra que la cantidad de accidentes y la tasa de mortalidad deben interpretarse juntas. Este patrón respalda incluir día de semana y fin de semana como características separadas.
H5: La modalidad más frecuente fue DESPISTE con 3806 accidentes (fig06), mientras que la mayor tasa de mortalidad fue ATROPELLO con 42.9% (fig07). Esto anticipa que MODALIDAD debe ser una variable dominante.
H6: LIMA concentra la mayor cantidad de accidentes (1553) (fig08), pero LA LIBERTAD lidera la tasa de mortalidad con 22.3% entre departamentos con al menos 30 casos (fig09). El ranking por volumen no equivale al ranking por severidad.
H7: El código de vía con mayor número de accidentes fue PE-1N con 1485 registros (fig10). Esto respalda el uso de frecuencia de vía como feature, calculada solo en train para evitar fuga.
H8: La distribución de kilómetros en las tres vías con más accidentes (fig11) muestra concentración por tramos, no una dispersión homogénea. Esto justifica conservar KILOMETRO como variable numérica con imputación controlada.
H9: El mapa mes × hora (fig12) evidencia patrones combinados de temporalidad. Por eso no basta con incluir fecha u hora cruda: se derivan mes, franja y codificación cíclica.
H10: La columna con mayor porcentaje de faltantes en la base limpia fue HORA con 1.1% (fig13). La baja magnitud de faltantes permite usar imputaciones simples con flags en lugar de descartar masivamente datos.

## Interpretación por figura

- fig01: La distribución del target confirma que la clase no mortal domina el dataset. Por eso la accuracy puede ser engañosa y se priorizan métricas de la clase mortal.
- fig02: La serie mensual permite observar variaciones temporales en la ocurrencia de accidentes. Esto fundamenta las variables derivadas de fecha.
- fig03: La distribución horaria identifica momentos de mayor volumen de accidentes. Volumen alto no implica automáticamente mayor severidad.
- fig04: La tasa de mortalidad por hora separa riesgo relativo de cantidad de eventos. Esta diferencia justifica nocturno, franja y codificación cíclica.
- fig05: El día de semana combina volumen y severidad en una misma lectura. La comparación sostiene usar día_semana y fin_de_semana como señales distintas.
- fig06: La modalidad muestra la composición del tipo de accidente. Esta variable describe la mecánica del siniestro y tiene relevancia directa para severidad.
- fig07: La tasa de mortalidad por modalidad muestra que algunas modalidades tienen riesgo relativo mayor aunque no sean las más frecuentes. Es uno de los hallazgos fuertes del EDA.
- fig08: El ranking por departamento muestra concentración territorial del volumen. Sirve para contextualizar la siniestralidad por carga de eventos.
- fig09: El ranking por tasa de mortalidad cambia la lectura territorial. Permite distinguir dónde ocurren más accidentes de dónde son relativamente más letales.
- fig10: Los códigos de vía más frecuentes sugieren concentración en rutas específicas. La frecuencia de vía se incorpora como señal sin tratar el código como número ordinal.
- fig11: La distribución por kilómetro sugiere tramos con concentración de accidentes en vías principales. Esto justifica no descartar KILOMETRO.
- fig12: El mapa mes × hora muestra patrones temporales cruzados. Refuerza que las features temporales deben capturar ciclos y no solo valores crudos.
- fig13: La matriz de faltantes transparenta la calidad de datos usada. La magnitud observada permite imputación con flags sin pérdida masiva de información.
