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
| `data/raw/BBDD_ONSV_VEHICULOS_2021-2025.xlsx` | base companion oficial de vehículos involucrados (v2) |
| `data/raw/BBDD_ONSV_PERSONAS_2021-2025.xlsx` | base companion oficial de personas involucradas (v2); columnas de desenlace vetadas |
| `data/raw/Formato_2_Diccionario_de_datos.docx` | diccionario de la fuente Sutran usada en la primera iteración; se conserva como evidencia histórica. El diccionario vigente de la fuente ONSV son los encabezados de la hoja `SINIESTROS` y el mapeo `COLUMN_MAP` de `src/block_b_dataset_audit.py` |
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

`src/model_protocol.py` produce 175 variables `float32` en orden fijo. Incluye contexto cronológico cíclico, ubicación, clase notificada, condiciones de vía y clima, cuatro interacciones predeclaradas y, desde la versión 2, doce agregados de escena de las bases companion (conteos y tipos de vehículos, personas involucradas, pasajeros, peatones, conductores fugados y edad media con su indicador de faltante). Excluye fallecidos, lesionados, vehículos dañados, causas investigadas, gravedad por persona, lugares de atención/defunción, dosajes etílicos, identificadores y campos con disponibilidad inadecuada.

### 3.3 Selección de red

La búsqueda cerrada compara tres arquitecturas y tres semillas:

| ID | Capas ocultas | Dropout | L2 | Learning rate |
|---|---|---:|---:|---:|
| `MLP_32_16` | 32-16 | 0.25 | 0.0001 | 0.001 |
| `MLP_64_32` | 64-32 | 0.35 | 0.0003 | 0.0005 |
| `MLP_32` | 32 | 0.20 | 0.0001 | 0.0005 |

Cada corrida usa Adam, entropía cruzada binaria, pesos de clase, reducción de tasa de aprendizaje y parada temprana por PR-AUC. La regla de selección es: mayor mediana de PR-AUC entre semillas, luego mayor mediana de F1 y finalmente menor IQR de PR-AUC. Con el contrato v2 la solución elegida es `MLP_32_16`, semilla 314 (la v1, con 162 features, había elegido `MLP_64_32`).

### 3.4 Calibración y umbrales

- Umbral crudo: 0.80, elegido en 2023 por máximo F1 con desempates predeclarados.
- Calibración: comparación OOF estratificada de cinco particiones dentro de 2023.
- Método desplegado: Platt por menor Brier OOF.
- Umbral calibrado: 0.30, elegido sobre probabilidades OOF de 2023.

Las escalas cruda y calibrada no se mezclan.

## 4. Resultados definitivos

### 4.1 MLP en referencia 2024-2025

| Métrica | Cruda (t=0.80) | Calibrada (t=0.30) |
|---|---:|---:|
| F1 multifatal | 0.4957 | 0.5058 |
| Precisión | 0.4715 | 0.4452 |
| Recall | 0.5225 | 0.5856 |
| PR-AUC | 0.4416 | 0.4416 |
| ROC-AUC | 0.8841 | 0.8841 |
| Brier | 0.1081 | 0.0683 |
| ECE | 0.1044 | 0.0169 |

La versión 1 (162 features, `MLP_64_32`) obtuvo PR-AUC 0.2249 y ROC-AUC 0.7482 sobre la misma referencia; la evaluación v2 es la segunda consulta declarada a ese periodo, con todo el diseño congelado en 2021-2023.

### 4.2 Comparación honesta

Bootstrap pareado sobre predicciones congeladas (2 000 remuestreos): la MLP supera a la regresión logística con significación estadística en ROC-AUC (Δ+0.026, IC 95% [+0.012, +0.042]) y sin significación en PR-AUC y F1. Frente al Random Forest, el bosque tiene mejor ranking nominal (PR-AUC 0.4704 vs 0.4416) sin significación en ninguna métrica. Esta evidencia justifica una red con no linealidad medible sobre el modelo lineal, sin afirmar dominancia sobre los ensambles de árboles.

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
