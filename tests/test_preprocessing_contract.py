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
                "FECHA": "2022-08-15",
                "HORA": "23:30",
                "DEPARTAMENTO": "PUNO",
                "ZONA": "RURAL",
                "RED_VIAL": "NACIONAL",
                "TIPO_VIA": "CARRETERA",
                "CODIGO_VIA": "PE-999X",
                "CLASE": "ATROPELLO FUGA",
                "CLIMA": "LLUVIOSO",
                "CARACTERISTICA_VIA": "CURVA",
                "PERFIL_VIA": "INCLINADA",
                "SUPERFICIE": "TROCHA",
                "LATITUD": None,
                "LONGITUD": None,
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
    assert "coord_faltante" in transformed.columns
    assert derived["via_freq"].iloc[0] == 0
    assert transformed["coord_faltante"].iloc[0] == 1
    assert transformed["clase_atropello"].iloc[0] == 1
    assert transformed["clima_lluvioso"].iloc[0] == 1
    assert transformed["superficie_trocha"].iloc[0] == 1
    assert transformed["zona_rural"].iloc[0] == 1


if __name__ == "__main__":
    test_preparar_entrada_contract()
    print("preprocessing-contract-ok")
