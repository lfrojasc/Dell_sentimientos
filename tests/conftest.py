import os
import sys
import tempfile
from pathlib import Path

# El registro de predicciones de las pruebas va a una carpeta temporal, no a registros/.
os.environ.setdefault("DELL_RUTA_LOG", str(Path(tempfile.mkdtemp(prefix="dell_pruebas_")) / "predicciones.sqlite"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import configuracion as cfg  # noqa: E402


def bert_disponible() -> bool:
    return cfg.USAR_BERT and (cfg.DIR_BERT / "model.safetensors").exists()


requiere_bert = pytest.mark.skipif(not bert_disponible(), reason="BERT no está descargado o está desactivado")
requiere_csv = pytest.mark.skipif(not cfg.RUTA_DATOS.exists(), reason="no está el CSV de datos")


@pytest.fixture(scope="session")
def servicio():
    from app.api import obtener_servicio
    return obtener_servicio()
