# Procedencia y disponibilidad de los datos

La entrega usa cuatro fuentes oficiales locales declaradas en
`data/raw/source_manifest.json`. Antes de leer un libro, el pipeline verifica
el conjunto exacto de archivos, su tamaño y SHA-256. Un archivo faltante,
adicional o modificado detiene la ejecución.

## Fuente y alcance

- Catálogo oficial: https://www.onsv.gob.pe/datosabiertos
- `SINIESTROS`: registro principal y fuente de `FALLECIDOS`.
- `VEHICULOS`: agregados de vehículos del registro consolidado. Su uso solo se
  defiende para clasificación histórica retrospectiva: no hay timestamps por
  campo que prueben disponibilidad al notificar.
- `PERSONAS`: **no se usa como entrada**. Solo se abre en una auditoría de
  linaje que demuestra que el número de filas y el conteo `GRAVEDAD=FALLECIDO`
  contienen el desenlace.
- Diccionario: referencia semántica de los campos.

Los libros 2024-2025 son preliminares. La fuente no publica timestamps de
disponibilidad por campo; por eso el proyecto no afirma inferencia en tiempo
real ni disponibilidad al momento de notificación.

## Prueba ejecutable del proxy

`src/source_provenance.py` contrasta, por código de siniestro, `FALLECIDOS` con
el conteo de personas cuya `GRAVEDAD` es `FALLECIDO`, y reconstruye
`target_multifatal`. Los resultados se persisten en
`report/tables/personas_target_proxy_identity.{csv,json}`. Esta evidencia es la
razón de excluir **todos** los agregados derivados de PERSONAS, no solamente la
columna `GRAVEDAD`.

## Independencia científica

La versión del artefacto es definitiva para la entrega, pero no existe una
cohorte externa o prospectiva intacta. Las etiquetas 2024-2025 ya fueron
consultadas y solo se conservan como referencia histórica provisional. Una
confirmación independiente exige datos futuros o externos no observados.
