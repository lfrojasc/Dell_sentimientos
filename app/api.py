"""API REST (FastAPI) + app de demostración (Gradio) en un solo servicio.

Arranque local:
    uvicorn app.api:app --port 8000

- Demo:          http://localhost:8000/demo
- Documentación: http://localhost:8000/docs  (se puede probar cada endpoint desde el navegador)
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # sin telemetría de Gradio

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import __version__
from . import configuracion as cfg
from .modelos import ErrorModelo, Servicio
from .registro import RegistroPredicciones, resumen_monitoreo

Modelo = Literal["baseline", "bert", "ambos"]


@lru_cache(maxsize=1)
def obtener_servicio() -> Servicio:
    return Servicio()


@lru_cache(maxsize=1)
def obtener_registro() -> RegistroPredicciones:
    return RegistroPredicciones()


def clasificar(textos: list[str], modelo: str, origen: str) -> list[list]:
    try:
        resultados = obtener_servicio().predecir(textos, modelo)
    except ErrorModelo as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    obtener_registro().guardar(textos, resultados, origen=origen)
    return resultados


# ----------------------------- Esquemas -----------------------------
class PeticionTweet(BaseModel):
    texto: str = Field(..., min_length=1, max_length=cfg.LIMITE_CARACTERES,
                       examples=["@Dell my new laptop stopped working after two days, worst support ever"])
    modelo: Modelo = Field("baseline", description="baseline (principal), bert (retador) o ambos")


class PeticionLote(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=cfg.LIMITE_LOTE)
    modelo: Modelo = "baseline"


class ResultadoModelo(BaseModel):
    modelo: str
    emocion: str = Field(description="Emoción final tras la política de decisión")
    emocion_mas_probable: str
    confianza: float = Field(description="Probabilidad de la emoción más probable")
    probabilidades: dict[str, float]
    revision_humana: bool = Field(description="True si el tweet debe revisarlo una persona")
    prioridad: bool = Field(description="True si la emoción es anger o disgust (atender primero)")
    motivos: list[str] = Field(description="Por qué va a revisión humana, si aplica")
    latencia_ms: float
    dominio: dict | None = Field(None, description="Resultado del filtro de idioma y tema")


class RespuestaTweet(BaseModel):
    resultados: list[ResultadoModelo]


class RespuestaLote(BaseModel):
    resultados: list[list[ResultadoModelo]]


# ----------------------------- Aplicación -----------------------------
@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    s = obtener_servicio()   # carga y verifica los modelos al arrancar, no en la primera consulta
    # Calentamiento (no se registra): inicializa spaCy y PyTorch para que la primera consulta real sea rápida.
    s.predecir(["My Dell laptop works fine"], "ambos" if s.bert is not None else "baseline")
    obtener_registro()
    yield


app = FastAPI(
    title="Emociones en tweets de Dell · FLAJ S.A.S.",
    version=__version__,
    description="Clasifica tweets de clientes de Dell en 8 emociones y marca los que necesitan revisión humana. "
                "Modelo principal: baseline TF-IDF + regresión logística. Modelo retador: BERT.",
    lifespan=ciclo_de_vida,
)


@app.get("/", include_in_schema=False)
def inicio():
    return RedirectResponse(url="/demo")


@app.get("/salud", tags=["servicio"])
def salud():
    s = obtener_servicio()
    return {"estado": "ok", "version": __version__, "baseline": True, "bert": s.bert is not None,
            "detalle_bert": s.error_bert}


@app.get("/modelo", tags=["servicio"])
def modelo():
    """Versiones, umbrales, métricas de prueba y huellas, tal como las registró el notebook."""
    reg = cfg.registro_modelo()
    s = obtener_servicio()
    return {
        "modelo_principal": cfg.parametros_baseline()["nombre"],
        "modelo_retador": cfg.parametros_bert()["nombre"],
        "bert_disponible": s.bert is not None,
        "politica": {"baseline": {"umbral_abstencion": s.baseline.umbral_abstencion,
                                  "umbrales_clase": s.baseline.umbrales_clase},
                     "bert": {"umbral_abstencion": cfg.parametros_bert()["umbral_abstencion"],
                              "filtro_idioma_tema": True}},
        "clases": cfg.CLASES,
        "clases_clave": cfg.clases_clave(),
        "metricas_prueba": reg["metricas_prueba"],
        "versiones_entrenamiento": reg["versiones"],
        "artefactos_sha256": reg["artefactos_sha256"],
    }


@app.post("/predecir", response_model=RespuestaTweet, tags=["predicción"])
def predecir(peticion: PeticionTweet):
    resultados = clasificar([peticion.texto], peticion.modelo, origen="api")
    return {"resultados": [p.como_dict() for p in resultados[0]]}


@app.post("/predecir_lote", response_model=RespuestaLote, tags=["predicción"])
def predecir_lote(peticion: PeticionLote):
    if any(not t.strip() or len(t) > cfg.LIMITE_CARACTERES for t in peticion.textos):
        raise HTTPException(status_code=422, detail=f"Cada texto debe tener entre 1 y {cfg.LIMITE_CARACTERES} caracteres")
    resultados = clasificar(peticion.textos, peticion.modelo, origen="api")
    return {"resultados": [[p.como_dict() for p in r] for r in resultados]}


@app.get("/monitoreo", tags=["monitoreo"])
def monitoreo(dias: int = Query(7, ge=1, le=365, description="Días hacia atrás")):
    """Indicadores de los últimos días por modelo, con el PSI frente a la referencia de prueba."""
    filas = obtener_registro().leer(dias)
    return {"dias": dias, "modelos": resumen_monitoreo(filas)}


# La demo se monta al final para que no tape las rutas de la API.
from .demo import construir_demo  # noqa: E402

import gradio as gr  # noqa: E402

app = gr.mount_gradio_app(app, construir_demo(), path="/demo")
