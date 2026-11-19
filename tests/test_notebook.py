"""El despliegue debe dar las mismas predicciones que el notebook.

`datos/robustez_notebook.csv` tiene la tabla de robustez tal como quedó en la salida del notebook
ejecutado (celda 102): emoción y confianza (2 decimales) de cada modelo para 24 textos.
"""
import csv
from pathlib import Path

import pytest

from .conftest import requiere_bert

CASOS = list(csv.DictReader(open(Path(__file__).parent / "datos" / "robustez_notebook.csv", encoding="utf-8")))
TOL = 0.0051   # el notebook muestra 2 decimales


def test_baseline_igual_que_el_notebook(servicio):
    prob = servicio.baseline.probabilidades([c["texto"] for c in CASOS])
    for c, p in zip(CASOS, prob):
        assert servicio.baseline.clases[p.argmax()] == c["baseline_emocion"], c["texto"]
        assert abs(p.max() - float(c["baseline_confianza"])) <= TOL, c["texto"]


@requiere_bert
def test_bert_igual_que_el_notebook(servicio):
    prob = servicio.bert.probabilidades([c["texto"] for c in CASOS])
    for c, p in zip(CASOS, prob):
        assert servicio.bert.clases[p.argmax()] == c["bert_emocion"], c["texto"]
        assert abs(p.max() - float(c["bert_confianza"])) <= TOL, c["texto"]


@requiere_bert
def test_bert_no_clasifica_lo_que_el_notebook_mostro_fuera_de_dominio(servicio):
    fuera = [c["texto"] for c in CASOS if c["tipo"] == "fuera de dominio"]
    for p in servicio.bert.predecir(fuera):
        assert p.emocion == "sin clasificar" and p.revision_humana
