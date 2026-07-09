from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import SEED
from src.preprocessing import fit_preprocessing_artifacts, preparar_entrada, save_artifacts


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "report" / "tables"

BASE_PATH = PROCESSED_DIR / "base_limpia.parquet"

# Outcome counts, post-investigation causes, and sparse signal columns stay in
# the clean base for EDA but never enter the model matrix.
EXCLUDED_COLUMNS = [
    "target_multifatal",
    "FALLECIDOS",
    "LESIONADOS",
    "VEHICULOS_DANADOS",
    "CAUSA_FACTOR",
    "CAUSA_ESPECIFICA",
    "SENAL_VERTICAL",
    "SENAL_HORIZONTAL",
    "CODIGO_SINIESTRO",
    "PROVINCIA",
    "DISTRITO",
    "ZONIFICACION",
]


def split_dataset(base: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    y = base["target_multifatal"].astype("int8")
    X = base.drop(columns=[column for column in EXCLUDED_COLUMNS if column in base.columns])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        stratify=y_temp,
        random_state=SEED,
    )
    return {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_demo_cases(X_val_raw: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    val = X_val_raw.copy()
    val["target_multifatal"] = y_val
    base_candidates = val[
        (val["target_multifatal"] == 0)
        & (val["hora_entera"].between(6, 17))
        & (val["CLASE"].isin(["DESPISTE", "CHOQUE"]))
        & (val["DEPARTAMENTO"].notna())
        & (val["ZONA"].notna())
        & (val["LATITUD"].notna())
    ]
    base = base_candidates.iloc[0] if not base_candidates.empty else val.iloc[0]

    def case_from_base(case_id: str, description: str, expected: str, reason: str, **overrides: object) -> dict[str, object]:
        row = {
            "caso_id": case_id,
            "descripcion": description,
            "FECHA": pd.to_datetime(base["FECHA"]).date().isoformat(),
            "HORA": base["HORA"],
            "DEPARTAMENTO": base["DEPARTAMENTO"],
            "ZONA": base["ZONA"],
            "RED_VIAL": base["RED_VIAL"],
            "TIPO_VIA": base["TIPO_VIA"],
            "CODIGO_VIA": base["CODIGO_VIA"],
            "CLASE": base["CLASE"],
            "CLIMA": base["CLIMA"],
            "CARACTERISTICA_VIA": base["CARACTERISTICA_VIA"],
            "PERFIL_VIA": base["PERFIL_VIA"],
            "SUPERFICIE": base["SUPERFICIE"],
            "LATITUD": base["LATITUD"],
            "LONGITUD": base["LONGITUD"],
            "esperado_cualitativo": expected,
            "motivo": reason,
        }
        row.update(overrides)
        return row

    return pd.DataFrame(
        [
            case_from_base(
                "demo_01_tipico_letalidad_simple",
                "Caso diurno de referencia con un solo fallecido tomado de validación.",
                "Probabilidad base baja o moderada.",
                "Referencia estable para comparar cambios controlados.",
            ),
            case_from_base(
                "demo_02_choque_nocturno_carretera",
                "Mismo caso base en horario nocturno.",
                "Debe subir la probabilidad frente al caso base.",
                "La nocturnidad concentra siniestros de alta letalidad.",
                HORA="23:30",
            ),
            case_from_base(
                "demo_03_curva_lluviosa",
                "Mismo caso base en curva con lluvia.",
                "Debe cambiar la probabilidad de forma razonable.",
                "Curva y lluvia degradan control y visibilidad.",
                CARACTERISTICA_VIA="CURVA",
                CLIMA="LLUVIOSO",
            ),
            case_from_base(
                "demo_04_codigo_no_visto",
                "Mismo caso base con código de vía no visto.",
                "La app no debe romperse y debe mapear via_freq=0.",
                "Verifica robustez ante categorías nuevas.",
                CODIGO_VIA="PE-999X",
            ),
            case_from_base(
                "demo_05_puno_rural",
                "Mismo caso base ubicado en Puno rural, red nacional.",
                "Debe confirmar que el encoding geográfico acepta PUNO.",
                "Puno es requisito explícito de los casos demo.",
                DEPARTAMENTO="PUNO",
                ZONA="RURAL",
                RED_VIAL="NACIONAL",
                LATITUD=-15.84,
                LONGITUD=-70.02,
            ),
        ]
    )


def write_feature_table(feature_list: list[str]) -> None:
    origins = {
        "LATITUD": ("COORDENADAS", "numérica continua", "Imputación por mediana de train + StandardScaler fit solo en train"),
        "LONGITUD": ("COORDENADAS", "numérica continua", "Imputación por mediana de train + StandardScaler fit solo en train"),
        "via_freq": ("CODIGO_VIA", "numérica continua", "Frecuencia relativa aprendida solo en train"),
        "anio": ("FECHA", "numérica continua", "StandardScaler fit solo en train"),
    }
    prefixes = {
        "departamento_": ("DEPARTAMENTO", "categórica one-hot", "One-Hot con categorías observadas en train"),
        "region_": ("DEPARTAMENTO", "categórica derivada", "Mapeo fijo departamento a región natural + One-Hot"),
        "via_prefijo_": ("CODIGO_VIA", "categórica derivada", "Extracción de prefijo alfabético + One-Hot"),
        "franja_": ("HORA", "categórica temporal", "Agrupación en franja horaria + One-Hot"),
        "clase_": ("CLASE SINIESTRO", "categórica one-hot", "Normalización a categorías cerradas + One-Hot"),
        "zona_": ("ZONA", "categórica one-hot", "Normalización rural/urbana + One-Hot"),
        "red_vial_": ("RED VIAL", "categórica one-hot", "Normalización a categorías cerradas + One-Hot"),
        "tipo_via_": ("TIPO DE VÍA", "categórica one-hot", "Agrupación de tipos de vía + One-Hot"),
        "clima_": ("CONDICIÓN CLIMÁTICA", "categórica one-hot", "Agrupación de condiciones climáticas + One-Hot"),
        "caracteristica_": ("CARACTERÍSTICAS DE VÍA", "categórica one-hot", "Agrupación recto/curva/intersección/estructura + One-Hot"),
        "perfil_": ("PERFIL LONGITUDINAL", "categórica one-hot", "Normalización plana/inclinada + One-Hot"),
        "superficie_": ("SUPERFICIE DE CALZADA", "categórica one-hot", "Agrupación pavimentada/afirmado/trocha + One-Hot"),
    }

    rows: list[dict[str, str]] = []
    for feature in feature_list:
        if feature in origins:
            origen, tipo, transformacion = origins[feature]
        else:
            for prefix, meta in prefixes.items():
                if feature.startswith(prefix):
                    origen, tipo, transformacion = meta
                    break
            else:
                origen, tipo, transformacion = (
                    "FECHA / HORA / COORDENADAS",
                    "derivada numérica/binaria",
                    "Derivación determinística sin fit",
                )
        rows.append(
            {
                "feature": feature,
                "origen": origen,
                "tipo": tipo,
                "transformacion": transformacion,
                "justificacion": "Variable pre-impacto disponible al momento del siniestro, sin fuga de resultado.",
            }
        )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "tab02_feature_contract.csv", index=False)


def run_block_d() -> dict[str, object]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(BASE_PATH)
    splits = split_dataset(base)

    scaler, encoders = fit_preprocessing_artifacts(splits["X_train_raw"])  # type: ignore[arg-type]
    X_train = preparar_entrada(splits["X_train_raw"], scaler=scaler, encoders=encoders)  # type: ignore[arg-type]
    X_val = preparar_entrada(splits["X_val_raw"], scaler=scaler, encoders=encoders)  # type: ignore[arg-type]
    X_test = preparar_entrada(splits["X_test_raw"], scaler=scaler, encoders=encoders)  # type: ignore[arg-type]

    save_artifacts(MODELS_DIR, scaler, encoders)
    (MODELS_DIR / "feature_list.json").write_text(
        json.dumps(encoders["feature_list"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_val.to_parquet(PROCESSED_DIR / "X_val.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
    splits["y_train"].rename("target_multifatal").to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)  # type: ignore[union-attr]
    splits["y_val"].rename("target_multifatal").to_frame().to_parquet(PROCESSED_DIR / "y_val.parquet", index=False)  # type: ignore[union-attr]
    splits["y_test"].rename("target_multifatal").to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)  # type: ignore[union-attr]

    demo_cases = build_demo_cases(splits["X_val_raw"], splits["y_val"])  # type: ignore[arg-type]
    demo_cases.to_csv(PROCESSED_DIR / "demo_cases.csv", index=False)

    write_feature_table(encoders["feature_list"])

    split_summary = {
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(len(encoders["feature_list"])),
        "train_multifatal_percentage": float(splits["y_train"].mean() * 100),  # type: ignore[union-attr]
        "val_multifatal_percentage": float(splits["y_val"].mean() * 100),  # type: ignore[union-attr]
        "test_multifatal_percentage": float(splits["y_test"].mean() * 100),  # type: ignore[union-attr]
        "demo_cases": int(demo_cases.shape[0]),
    }
    (TABLES_DIR / "tab02_split_summary_block_d.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([split_summary]).to_csv(TABLES_DIR / "tab02_split_summary_block_d.csv", index=False)
    return split_summary


if __name__ == "__main__":
    print(json.dumps(run_block_d(), ensure_ascii=False, indent=2))
