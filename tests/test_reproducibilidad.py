"""Reconstruye la partición de prueba desde el CSV y compara las métricas con el notebook.

La parte de BERT clasifica 4.824 tweets (unos 10 a 15 minutos en CPU), así que solo corre si se
define la variable de entorno DELL_PRUEBA_LENTA=1.
"""
import os

import numpy as np
import pytest

from app import configuracion as cfg

from .conftest import requiere_bert, requiere_csv


@pytest.fixture(scope="module")
def particion():
    from scripts.reproducir_prueba import preparar_datos
    return preparar_datos(cfg.RUTA_DATOS)


def _comparar(obtenidas, esperadas):
    for k, v in obtenidas.items():
        assert round(v, 4) == pytest.approx(esperadas[k], abs=1e-4), k


@requiere_csv
def test_particion_igual_que_el_notebook(particion):
    df_tr, df_te = particion
    assert (len(df_tr), len(df_te)) == (19295, 4824)
    assert not set(df_tr["Texto base"]) & set(df_te["Texto base"])


@requiere_csv
def test_metricas_baseline_igual_que_el_notebook(servicio, particion):
    from scripts.reproducir_prueba import metricas
    _, df_te = particion
    b = servicio.baseline
    prob = b.probabilidades(df_te["Text"].tolist())
    pred = np.array(cfg.CLASES)[prob.argmax(axis=1)]
    _comparar(metricas(df_te["emotion"].values, pred, prob), cfg.registro_modelo()["metricas_prueba"][b.nombre])


@requiere_csv
@requiere_bert
@pytest.mark.skipif(os.environ.get("DELL_PRUEBA_LENTA") != "1", reason="prueba lenta: definir DELL_PRUEBA_LENTA=1")
def test_metricas_bert_igual_que_el_notebook(servicio, particion):
    from scripts.reproducir_prueba import metricas
    _, df_te = particion
    t = servicio.bert
    prob = t.probabilidades(df_te["Text"].tolist())
    pred = np.array(cfg.CLASES)[prob.argmax(axis=1)]
    _comparar(metricas(df_te["emotion"].values, pred, prob), cfg.registro_modelo()["metricas_prueba"][t.nombre])
