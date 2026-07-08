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


def split_dataset(base: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    y = base["target_mortal"].astype("int8")
    X = base.drop(columns=["target_mortal", "FALLECIDOS", "HERIDOS"])

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
    val["target_mortal"] = y_val
    base_candidates = val[
        (val["target_mortal"] == 0)
        & (val["hora_entera"].between(6, 17))
        & (val["MODALIDAD"].isin(["DESPISTE", "CHOQUE"]))
        & (val["DEPARTAMENTO"].notna())
        & (val["CODIGO_VIA"].notna())
    ]
    base = base_candidates.iloc[0] if not base_candidates.empty else val.iloc[0]

    def case_from_base(case_id: str, description: str, expected: str, reason: str, **overrides: object) -> dict[str, object]:
        row = {
            "caso_id": case_id,
            "descripcion": description,
            "FECHA": pd.to_datetime(base["FECHA"]).date().isoformat(),
            "HORA": base["HORA"],
            "DEPARTAMENTO": base["DEPARTAMENTO"],
            "CODIGO_VIA": base["CODIGO_VIA"],
            "KILOMETRO": base["KILOMETRO"],
            "MODALIDAD": base["MODALIDAD"],
            "esperado_cualitativo": expected,
            "motivo": reason,
        }
        row.update(overrides)
        return row

    return pd.DataFrame(
        [
            case_from_base(
                "demo_01_tipico_no_mortal",
                "Caso diurno de referencia no mortal tomado de validación.",
                "Probabilidad base baja o moderada.",
                "Referencia estable para comparar cambios controlados.",
            ),
            case_from_base(
                "demo_02_atropello",
                "Mismo caso base cambiando solo la modalidad a atropello.",
                "Debe subir la probabilidad frente al caso base.",
                "Atropello suele asociarse a mayor letalidad.",
                MODALIDAD="ATROPELLO",
            ),
            case_from_base(
                "demo_03_nocturno",
                "Mismo caso base en horario nocturno.",
                "Debe cambiar la probabilidad de forma razonable.",
                "La nocturnidad afecta visibilidad, fatiga y riesgo.",
                HORA="23:30",
            ),
            case_from_base(
                "demo_04_codigo_no_visto",
                "Mismo caso base con código de vía no visto.",
                "La app no debe romperse y debe mapear via_freq=0.",
                "Verifica robustez ante categorías nuevas.",
                CODIGO_VIA="PE-999X",
            ),
            case_from_base(
                "demo_05_puno",
                "Mismo caso base ubicado en Puno.",
                "Debe confirmar que el encoding geográfico acepta PUNO.",
                "Puno es requisito explícito de los casos demo.",
                DEPARTAMENTO="PUNO",
            ),
        ]
    )


def write_feature_table(feature_list: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for feature in feature_list:
        if feature in {"KILOMETRO", "via_freq", "anio"}:
            rows.append(
                {
                    "feature": feature,
                    "origen": "KILOMETRO / CODIGO_VIA / FECHA",
                    "tipo": "numérica continua",
                    "transformacion": "Imputación/encoding en train y StandardScaler fit solo en train",
                    "justificacion": "Escala compatible con MLP y sin fuga de validación/test.",
                }
            )
        elif feature.startswith("departamento_"):
            rows.append(
                {
                    "feature": feature,
                    "origen": "DEPARTAMENTO",
                    "tipo": "categórica one-hot",
                    "transformacion": "One-Hot con categorías observadas en train",
                    "justificacion": "Captura señal geográfica sin ordinalidad artificial.",
                }
            )
        elif feature.startswith("region_"):
            rows.append(
                {
                    "feature": feature,
                    "origen": "DEPARTAMENTO",
                    "tipo": "categórica derivada",
                    "transformacion": "Mapeo fijo departamento a región natural + One-Hot",
                    "justificacion": "Agrupa patrones geográficos generales para reducir dispersión.",
                }
            )
        elif feature.startswith("via_prefijo_"):
            rows.append(
                {
                    "feature": feature,
                    "origen": "CODIGO_VIA",
                    "tipo": "categórica derivada",
                    "transformacion": "Extracción de prefijo alfabético + One-Hot",
                    "justificacion": "Representa familias de vías sin depender del código completo.",
                }
            )
        elif feature.startswith("franja_"):
            rows.append(
                {
                    "feature": feature,
                    "origen": "HORA",
                    "tipo": "categórica temporal",
                    "transformacion": "Agrupación en franja horaria + One-Hot",
                    "justificacion": "Modela periodos del día interpretables.",
                }
            )
        elif feature.startswith("modalidad_"):
            rows.append(
                {
                    "feature": feature,
                    "origen": "MODALIDAD",
                    "tipo": "categórica one-hot",
                    "transformacion": "Normalización a categorías cerradas + One-Hot",
                    "justificacion": "La mecánica del accidente impacta directamente la severidad.",
                }
            )
        else:
            rows.append(
                {
                    "feature": feature,
                    "origen": "FECHA / HORA / KILOMETRO",
                    "tipo": "derivada numérica/binaria",
                    "transformacion": "Derivación determinística sin fit",
                    "justificacion": "Aporta señal temporal, horaria o de faltantes sin fuga.",
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
    splits["y_train"].rename("target_mortal").to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)  # type: ignore[union-attr]
    splits["y_val"].rename("target_mortal").to_frame().to_parquet(PROCESSED_DIR / "y_val.parquet", index=False)  # type: ignore[union-attr]
    splits["y_test"].rename("target_mortal").to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)  # type: ignore[union-attr]

    demo_cases = build_demo_cases(splits["X_val_raw"], splits["y_val"])  # type: ignore[arg-type]
    demo_cases.to_csv(PROCESSED_DIR / "demo_cases.csv", index=False)

    write_feature_table(encoders["feature_list"])

    split_summary = {
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(len(encoders["feature_list"])),
        "train_mortal_percentage": float(splits["y_train"].mean() * 100),  # type: ignore[union-attr]
        "val_mortal_percentage": float(splits["y_val"].mean() * 100),  # type: ignore[union-attr]
        "test_mortal_percentage": float(splits["y_test"].mean() * 100),  # type: ignore[union-attr]
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
