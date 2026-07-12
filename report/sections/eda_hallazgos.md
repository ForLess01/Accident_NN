# Hallazgos del Bloque C — EDA

H1: La clase multifatal (2+ fallecidos) representa 10.2% de los siniestros fatales (fig01). Esto confirma el desbalance y justifica evaluar con F1, recall y PR-AUC, no con accuracy aislada.
H2: El mes con más siniestros fatales fue 2021-06 con 229 registros (fig02). La serie mensual muestra que la frecuencia no es constante, por lo que las variables temporales aportan señal al modelo.
H3: La hora con más siniestros fue las 19:00 con 667 registros (fig03). En cambio, la mayor tasa multifatal por hora aparece a las 14:00 con 16.3% (fig04), mostrando que volumen y letalidad no son lo mismo.
H4: La comparación por día de semana (fig05) muestra que la cantidad de siniestros y la tasa multifatal deben interpretarse juntas. Este patrón respalda incluir día de semana y fin de semana como características separadas.
H5: La clase de siniestro más frecuente fue CHOQUE con 3260 casos (fig06), mientras que la mayor tasa multifatal fue DESPISTE con 17.7% (fig07). Esto anticipa que la clase de siniestro debe ser una variable dominante.
H6: LIMA concentra la mayor cantidad de siniestros fatales (1917) (fig08), pero HUANCAVELICA lidera la tasa multifatal con 28.0% entre departamentos con al menos 30 casos (fig09). El ranking por volumen no equivale al ranking por letalidad.
H7: El código de carretera con mayor número de siniestros fue PE-1N con 1032 registros (fig10). Esto respalda el uso de frecuencia de vía como feature, calculada solo en train para evitar fuga.
H8: Las condiciones iniciales separan el riesgo (fig11): zona rural 15.1% vs urbana 3.6%, y clima lluvioso 17.6% vs despejado 9.8%. Estas variables aportan contexto relevante para el modelo.
H9: El mapa mes × hora (fig12) evidencia patrones combinados de temporalidad. Por eso no basta con incluir fecha u hora cruda: se derivan mes, franja y codificación cíclica.
H10: La columna con mayor porcentaje de faltantes en la base limpia fue SENAL_HORIZONTAL con 77.9% (fig13). Las señales viales se excluyen del modelo por ese motivo; el resto admite imputación simple con flags.
H11: El scatter geográfico (fig23) muestra los siniestros siguiendo los corredores viales del país, con eventos multifatales distribuidos en todas las regiones. Las coordenadas aportan señal espacial continua que complementa al departamento.

## Interpretación por figura

- fig01: La distribución del target confirma que los siniestros con un solo fallecido dominan. Por eso la accuracy puede ser engañosa y se priorizan métricas de la clase multifatal.
- fig02: La serie mensual permite observar variaciones temporales en la ocurrencia de siniestros fatales. Esto fundamenta las variables derivadas de fecha.
- fig03: La distribución horaria identifica momentos de mayor volumen de siniestros. Volumen alto no implica automáticamente mayor letalidad.
- fig04: La tasa multifatal por hora separa riesgo relativo de cantidad de eventos. Esta diferencia justifica nocturno, franja y codificación cíclica.
- fig05: El día de semana combina volumen y letalidad en una misma lectura. La comparación sostiene usar día_semana y fin_de_semana como señales distintas.
- fig06: La clase de siniestro muestra la composición de la mecánica del evento. Esta variable tiene relevancia directa para la letalidad.
- fig07: La tasa multifatal por clase muestra que algunas mecánicas tienen riesgo relativo mayor aunque no sean las más frecuentes. Es uno de los hallazgos fuertes del EDA.
- fig08: El ranking por departamento muestra concentración territorial del volumen. Sirve para contextualizar la siniestralidad por carga de eventos.
- fig09: El ranking por tasa multifatal cambia la lectura territorial. Permite distinguir dónde ocurren más siniestros de dónde son relativamente más letales.
- fig10: Los códigos de carretera más frecuentes sugieren concentración en rutas específicas. La frecuencia de vía se incorpora como señal sin tratar el código como número ordinal.
- fig11: Las tasas por zona, clima, característica y superficie son las variables pre-impacto nuevas de esta fuente. Separan el riesgo estructural del contexto del siniestro.
- fig12: El mapa mes × hora muestra patrones temporales cruzados. Refuerza que las features temporales deben capturar ciclos y no solo valores crudos.
- fig13: La matriz de faltantes transparenta la calidad de datos usada. Justifica excluir señales viales y usar imputación con flags en el resto.
- fig23: El scatter geográfico visualiza la red vial implícita en los datos. Las coordenadas entran al modelo como variables continuas estandarizadas.
