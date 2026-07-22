# Mapa de reproducción de la entrega definitiva

Este archivo no describe versiones temporales. La única entrega canónica es `models/final/` (`canonical-3.0.0`).

1. `source_provenance.py` verifica los cuatro archivos oficiales y prueba la identidad PERSONAS-target.
2. `block_b_dataset_audit.py` crea `base_limpia.parquet` con SINIESTROS y agregados VEHICULOS; PERSONAS no entra al dataset de modelado.
3. `block_e_modeling.py` selecciona arquitectura con 2021/2022 y reajusta una vez en 2021-2022.
4. `final_model_bundle.py` valida calibración y umbral en 2023 y conserva 2024-2025 como referencia histórica ya consultada.
5. `temporal_diagnostics.py` ejecuta dos folds rolling internos disjuntos por rol.
6. `final_explainability.py`, `demo_cases.py` y `final_evaluation_figures.py` generan evidencia derivada del bundle.
7. `build_report.py` compila y sella el PDF por hash; `check_release.py` aplica el gate final.

No existe test externo/prospectivo intacto. El siguiente avance científico válido requiere una cohorte futura o externa no observada.
