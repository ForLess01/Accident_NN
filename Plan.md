# Especificación definitiva de Accident_NN

Este documento es el contrato vigente del proyecto. Describe una sola solución académica reproducible y reemplaza cualquier planificación histórica.

## Estado

**Implementación completada en el workspace.** El contenido incluye la fuente ONSV, la base limpia, la MLP seleccionada, la calibración, la evaluación de referencia, la explicabilidad, la interfaz, las pruebas y el informe. El gate estricto confirma el estado de versionado antes de una entrega desde Git.

## 1. Problema y alcance

El objetivo es estimar `target_multifatal`, definido como:

```text
target_multifatal = 1 si FALLECIDOS >= 2; 0 en caso contrario
```

El universo ya está restringido a siniestros fatales registrados por el ONSV. Por tanto, la salida representa **multifatalidad condicional**, no mortalidad sobre todos los accidentes. El uso defendible es priorización posterior a la notificación.

## 2. Fuente de verdad

| Artefacto | Función |
|---|---|
| `data/raw/BBDD_ONSV_SINIESTROS_FATALES_2021-2025.xlsx` | extracción oficial original |
| `data/raw/Formato_2_Diccionario_de_datos.docx` | diccionario oficial |
| `data/processed/base_limpia.parquet` | base canónica de 9 104 registros |
| `models/final/manifest.json` | procedencia, hashes, particiones y métricas |
| `models/final/model_selection.json` | decisión de arquitectura, semilla y umbral crudo |
| `report/tables/final_reference_metrics_2024_2025.json` | métricas definitivas de referencia |

No hay fuentes alternativas ni artefactos de modelos descartados dentro del proyecto.

## 3. Protocolo científico

### 3.1 Particiones cronológicas

| Periodo | Rol | N | Positivos |
|---|---|---:|---:|
| 2021-2022 | entrenamiento | 4 872 | 510 |
| 2023 | selección de arquitectura, semilla, umbrales y calibración | 2 000 | 196 |
| 2024-2025 | referencia histórica | 2 232 | 222 |

La referencia no participa en entrenamiento, búsqueda, selección ni calibración. Sus etiquetas ya fueron observadas; no puede presentarse como un test nuevo ni reutilizarse para afinar el sistema.

### 3.2 Contrato de entrada

`src/model_protocol.py` produce 162 variables `float32` en orden fijo. Incluye contexto cronológico cíclico, ubicación, clase notificada, condiciones de vía y clima, y cuatro interacciones predeclaradas. Excluye fallecidos, lesionados, vehículos dañados, causas investigadas, identificadores y campos con disponibilidad inadecuada.

### 3.3 Selección de red

La búsqueda cerrada compara tres arquitecturas y tres semillas:

| ID | Capas ocultas | Dropout | L2 | Learning rate |
|---|---|---:|---:|---:|
| `MLP_32_16` | 32-16 | 0.25 | 0.0001 | 0.001 |
| `MLP_64_32` | 64-32 | 0.35 | 0.0003 | 0.0005 |
| `MLP_32` | 32 | 0.20 | 0.0001 | 0.0005 |

Cada corrida usa Adam, entropía cruzada binaria, pesos de clase, reducción de tasa de aprendizaje y parada temprana por PR-AUC. La regla de selección es: mayor mediana de PR-AUC entre semillas, luego mayor mediana de F1 y finalmente menor IQR de PR-AUC. La solución elegida es `MLP_64_32`, semilla 314.

### 3.4 Calibración y umbrales

- Umbral crudo: 0.65, elegido en 2023 por máximo F1 con desempates predeclarados.
- Calibración: comparación OOF estratificada de cinco particiones dentro de 2023.
- Método desplegado: Platt por menor Brier OOF.
- Umbral calibrado: 0.20, elegido sobre probabilidades OOF de 2023.

Las escalas cruda y calibrada no se mezclan.

## 4. Resultados definitivos

### 4.1 MLP en referencia 2024-2025

| Métrica | Cruda | Calibrada |
|---|---:|---:|
| F1 multifatal | 0.3030 | 0.2958 |
| Precisión | 0.2346 | 0.2359 |
| Recall | 0.4279 | 0.3964 |
| PR-AUC | 0.2249 | 0.2249 |
| ROC-AUC | 0.7482 | 0.7482 |
| Brier | 0.1817 | 0.0831 |
| ECE | 0.2623 | 0.0085 |

### 4.2 Comparación honesta

La MLP tiene el PR-AUC y ROC-AUC nominalmente más altos entre los modelos declarados. La regresión logística obtiene mayor F1, y Random Forest presenta menor F1. Esta evidencia justifica una red con señal útil y metodología sólida, pero no una afirmación de superioridad universal o estadísticamente significativa.

## 5. Interfaz

`app/streamlit_app.py` ofrece cinco secciones enlazables mediante `?section=`:

1. **Panorama**;
2. **Estimar**;
3. **Explorar datos**;
4. **Patrones regionales**;
5. **Evidencia del modelo**.

La app no entrena, no recalibra, no escribe artefactos y no duplica lógica de transformación. `src/app_inference.py` carga el bundle, verifica hashes y limpia resultados obsoletos cuando una entrada falla. El contrato restringe fechas a 2021-2025, valida el par de coordenadas con el GeoJSON del Perú, comprueba coherencia departamental, controla el formato de códigos de vía y enmascara subgrupos con soporte menor que 30. Las matrices y curvas tienen alternativas tabulares descargables.

## 6. Explicabilidad

`src/final_explainability.py` usa Gradient SHAP con fondo 2021-2022 y ejemplos de 2023. No carga etiquetas ni filas de 2024-2025. Los resultados globales se agrupan por variable original y se presentan como asociaciones del modelo, nunca como causalidad.

## 7. Criterios de aceptación

- [x] Fuente ONSV y diccionario preservados.
- [x] Base limpia y casos demo disponibles en un clon completo.
- [x] Corte cronológico sin fuga de resultado.
- [x] Búsqueda cerrada 3 x 3 y semilla representativa.
- [x] Baselines bajo la misma política de umbral.
- [x] Calibración seleccionada solo con 2023.
- [x] Referencia 2024-2025 bloqueada para nuevos ajustes.
- [x] Bundle único con hashes y paridad de inferencia.
- [x] Explicabilidad sin acceso al periodo de referencia.
- [x] Interfaz de solo lectura verificada con AppTest.
- [x] Formulario sin valores plausibles implícitos y caso demo de carga explícita.
- [x] Validación geográfica Perú/departamento y periodo académico 2021-2025.
- [x] Navegación enlazable y alternativas tabulares para evidencia gráfica.
- [x] Informe y defensa alineados con los artefactos.

## 8. Comandos de control

```bash
./scripts/check_release.py --local-content  # workspace en desarrollo
./scripts/check_release.py                  # entrega desde Git
```

El modelo solo puede cambiar mediante una nueva fuente o un periodo futuro verdaderamente no observado. No se autoriza mejorar métricas ajustando sobre 2024-2025.
