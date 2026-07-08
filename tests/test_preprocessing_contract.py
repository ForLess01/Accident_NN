from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import load_artifacts, preparar_entrada
from src.features import derive_base_features


def test_preparar_entrada_contract() -> None:
    feature_list = json.loads((ROOT / "models" / "feature_list.json").read_text())
    scaler, encoders = load_artifacts(ROOT / "models")

    sample = pd.DataFrame(
        [
            {
                "FECHA": "2021-08-15",
                "HORA": "23:30",
                "DEPARTAMENTO": "PUNO",
                "CODIGO_VIA": "PE-999X",
                "KILOMETRO": -5,
                "MODALIDAD": "ATROPELLO",
                "hora_entera": 23,
            }
        ]
    )

    transformed = preparar_entrada(sample, scaler=scaler, encoders=encoders)
    derived = derive_base_features(sample, via_frequency_map=encoders["via_frequency_map"])

    assert transformed.shape[1] == len(feature_list)
    assert transformed.columns.tolist() == feature_list
    assert transformed.isna().sum().sum() == 0
    assert "via_freq" in transformed.columns
    assert "km_faltante" in transformed.columns
    assert derived["via_freq"].iloc[0] == 0
    assert transformed["km_faltante"].iloc[0] == 1


if __name__ == "__main__":
    test_preparar_entrada_contract()
    print("preprocessing-contract-ok")
