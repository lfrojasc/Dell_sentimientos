import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.api import app, obtener_registro

from .conftest import bert_disponible


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok" and r.json()["baseline"] is True


def test_inicio_redirige_a_la_demo(cliente):
    r = cliente.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/demo"
    assert cliente.get("/demo/").status_code == 200


def test_modelo_muestra_el_registro(cliente):
    d = cliente.get("/modelo").json()
    assert d["modelo_principal"].startswith("Baseline")
    assert d["politica"]["baseline"]["umbrales_clase"] == {"anger": 0.25, "disgust": 0.20}


def test_predecir_baseline(cliente):
    r = cliente.post("/predecir", json={"texto": "Your customer service is horrible, worst experience ever"})
    assert r.status_code == 200
    (res,) = r.json()["resultados"]
    assert res["modelo"] == "baseline" and res["emocion"] == "anger" and res["prioridad"] is True
    assert abs(sum(res["probabilidades"].values()) - 1) < 1e-3


@pytest.mark.parametrize("cuerpo", [{"texto": ""}, {"texto": "x" * 1001}, {"texto": "hola", "modelo": "otro"}])
def test_validacion_de_entrada(cliente, cuerpo):
    assert cliente.post("/predecir", json=cuerpo).status_code == 422


def test_predecir_lote(cliente):
    r = cliente.post("/predecir_lote", json={"textos": ["I love my new Dell", "Dell support never solved my problem"]})
    assert r.status_code == 200 and len(r.json()["resultados"]) == 2


def test_bert_segun_disponibilidad(cliente):
    r = cliente.post("/predecir", json={"texto": "My new Dell laptop works perfectly", "modelo": "ambos"})
    if bert_disponible():
        assert [x["modelo"] for x in r.json()["resultados"]] == ["baseline", "bert"]
    else:
        assert r.status_code == 503


def test_registro_no_guarda_el_texto(cliente):
    secreto = "my phone number is 3001234567 and my laptop is broken"
    cliente.post("/predecir", json={"texto": secreto})
    con = sqlite3.connect(obtener_registro().ruta)
    columnas = [c[1] for c in con.execute("PRAGMA table_info(predicciones)")]
    contenido = " ".join(str(v) for fila in con.execute("SELECT * FROM predicciones") for v in fila)
    assert "texto" not in columnas and "3001234567" not in contenido


def test_monitoreo(cliente):
    for _ in range(30):
        cliente.post("/predecir", json={"texto": "Dell support never solved my problem"})
    d = cliente.get("/monitoreo", params={"dias": 1}).json()["modelos"]["baseline"]
    assert d["predicciones"] >= 30
    assert d["psi"] is not None and d["psi"] > 0.25   # 30 tweets iguales: la mezcla de emociones cambió mucho
    assert d["alerta"].startswith("cambio fuerte")
