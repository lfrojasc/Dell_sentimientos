import shutil

import numpy as np
import pytest

from app import configuracion as cfg
from app.modelos import ErrorModelo, ModeloBaseline, aplicar_umbrales

C = cfg.CLASES


def fila(**probs):
    p = np.zeros(len(C))
    for clase, v in probs.items():
        p[C.index(clase)] = v
    return p[None, :]


def test_parametros_salen_del_registro_del_notebook():
    pb, pt = cfg.parametros_baseline(), cfg.parametros_bert()
    assert pb["umbral_abstencion"] == 0.35
    assert pb["umbrales_clase"] == {"anger": 0.25, "disgust": 0.20}
    assert pt["umbral_abstencion"] == 0.45
    assert pt["max_len"] == 80
    assert cfg.clases_clave() == ["anger", "disgust"]


UMB = {"anger": 0.25, "disgust": 0.20}


def test_sin_alerta_gana_la_mas_probable():
    assert aplicar_umbrales(fila(joy=0.6, anger=0.2, disgust=0.1), UMB, C)[0] == "joy"


def test_anger_sobre_su_umbral_gana_aunque_no_sea_la_mas_probable():
    assert aplicar_umbrales(fila(joy=0.5, anger=0.26), UMB, C)[0] == "anger"


def test_con_dos_alertas_gana_la_mas_probable_de_ellas():
    assert aplicar_umbrales(fila(joy=0.4, anger=0.27, disgust=0.30), UMB, C)[0] == "disgust"


def test_abstencion_en_el_limite(servicio):
    b = servicio.baseline
    assert b.umbral_abstencion == 0.35
    # Confianza 0,32 (< 0,35): el texto del clima va a revisión humana, como en el notebook.
    p = b.predecir(["The weather in Bogota is sunny today"])[0]
    assert p.revision_humana and p.motivos[0].startswith("confianza baja")


def test_prioridad_para_anger_y_disgust(servicio):
    p = servicio.baseline.predecir(["Your customer service is horrible, worst experience ever"])[0]
    assert p.emocion == "anger" and p.prioridad and not p.revision_humana


def test_huella_distinta_impide_cargar_el_modelo(tmp_path):
    copia = tmp_path / "baseline.joblib"
    shutil.copy(cfg.RUTA_BASELINE, copia)
    with open(copia, "ab") as fh:
        fh.write(b"0")
    with pytest.raises(ErrorModelo, match="SHA-256"):
        ModeloBaseline(copia)
