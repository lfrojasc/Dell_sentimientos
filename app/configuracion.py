"""Rutas y parámetros del despliegue.

Los umbrales NO se escriben a mano: se leen de `artefactos/registro_modelo.json`, el archivo
que generó el notebook al final de su ejecución. Así el despliegue usa exactamente los mismos
valores que se reportaron en el informe.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_ARTEFACTOS = Path(os.environ.get("DELL_DIR_ARTEFACTOS", RAIZ / "artefactos"))
DIR_BERT = Path(os.environ.get("DELL_DIR_BERT", DIR_ARTEFACTOS / "bert_final"))
RUTA_BASELINE = DIR_ARTEFACTOS / "baseline.joblib"
RUTA_REGISTRO_MODELO = DIR_ARTEFACTOS / "registro_modelo.json"
RUTA_REFERENCIA_MONITOREO = DIR_ARTEFACTOS / "referencia_monitoreo.json"
RUTA_DATOS = Path(os.environ.get("DELL_RUTA_DATOS", RAIZ / "datos" / "sentiment-emotion-labelled_Dell_tweets.csv"))
RUTA_LOG = Path(os.environ.get("DELL_RUTA_LOG", RAIZ / "registros" / "predicciones.sqlite"))

# "0" desactiva BERT aunque esté descargado (por ejemplo, en la imagen de prueba de GitHub Actions).
USAR_BERT = os.environ.get("DELL_USAR_BERT", "1") == "1"

# Release de GitHub donde se publica BERT (438 MB no caben en el repositorio).
URL_BERT = os.environ.get(
    "DELL_URL_BERT",
    "https://github.com/lfrojasc/Dell_sentimientos/releases/download/v1.0/bert_final.zip",
)

# Orden de las clases: el mismo de LabelEncoder en el notebook (orden alfabético).
CLASES = ["anger", "anticipation", "disgust", "fear", "joy", "optimism", "sadness", "surprise"]

LIMITE_CARACTERES = 1000
LIMITE_LOTE = 100

# Umbrales de alerta del monitoreo (sección de gobierno del informe).
PSI_VIGILAR = 0.10
PSI_INVESTIGAR = 0.25
MIN_TWEETS_PSI = 30


@lru_cache(maxsize=1)
def registro_modelo() -> dict:
    with open(RUTA_REGISTRO_MODELO, encoding="utf-8") as fh:
        return json.load(fh)


def _entrada(prefijo: str) -> tuple[str, dict]:
    """Busca un modelo en el registro por el inicio de su nombre (los nombres son largos)."""
    for nombre, datos in registro_modelo()["modelos"].items():
        if nombre.startswith(prefijo):
            return nombre, datos
    raise KeyError(f"No hay un modelo que empiece por {prefijo!r} en {RUTA_REGISTRO_MODELO}")


def parametros_baseline() -> dict:
    nombre, datos = _entrada("Baseline")
    return {
        "nombre": nombre,
        "umbral_abstencion": float(datos["umbral_abstencion"]),
        "umbrales_clase": {k: float(v) for k, v in datos["umbrales_clase"].items()},
        "sha256": registro_modelo()["artefactos_sha256"]["baseline.joblib"],
    }


def parametros_bert() -> dict:
    nombre, datos = _entrada("BERT")
    return {
        "nombre": nombre,
        "umbral_abstencion": float(datos["umbral_abstencion"]),
        "max_len": int(datos["max_len"]),
        "sha256": registro_modelo()["artefactos_sha256"]["bert_final/"],
    }


def clases_clave() -> list[str]:
    return list(registro_modelo()["prerregistro"]["clases_clave"])
