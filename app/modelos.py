"""Carga de los modelos y política de decisión.

- **Baseline (modelo principal):** TF-IDF de lemas + regresión logística. Política recomendada en
  el informe: umbral por clase para anger y disgust, y abstención (revisión humana) cuando la
  emoción más probable no alcanza el umbral de confianza.
- **BERT (modelo retador):** bert-base-uncased ajustado. Política: emoción más probable,
  abstención con su propio umbral y, además, el filtro de idioma y tema.

Antes de usar un modelo se verifica su huella SHA-256 contra `registro_modelo.json`. Si el
archivo cambió (otra versión, archivo dañado), el servicio no arranca con ese modelo.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import configuracion as cfg
from .filtro import FiltroDominio, ResultadoFiltro
from .limpieza import preparar_baseline, preparar_bert


class ErrorModelo(RuntimeError):
    pass


def sha256_ruta(ruta: Path) -> str:
    """Misma función que usó el notebook (celda 114) para registrar las huellas."""
    ruta = str(ruta)
    h = hashlib.sha256()
    archivos = [ruta] if os.path.isfile(ruta) else sorted(
        os.path.join(r, f) for r, _, fs in os.walk(ruta) for f in fs)
    for f in archivos:
        with open(f, "rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
    return h.hexdigest()


def aplicar_umbrales(prob: np.ndarray, umbrales: dict[str, float], clases: list[str]) -> np.ndarray:
    """Copia de la celda 60 del notebook: emoción más probable, salvo que anger o disgust superen su
    umbral; entonces gana la más probable de las que lo superan."""
    base = prob.argmax(axis=1)
    puntaje = np.full(prob.shape, -1.0)
    for clase, th in umbrales.items():
        j = clases.index(clase)
        puntaje[:, j] = np.where(prob[:, j] >= th, prob[:, j], -1.0)
    hay_alerta = puntaje.max(axis=1) >= 0
    return np.array(clases)[np.where(hay_alerta, puntaje.argmax(axis=1), base)]


@dataclass
class Prediccion:
    modelo: str
    emocion: str
    emocion_mas_probable: str
    confianza: float
    probabilidades: dict[str, float]
    revision_humana: bool
    prioridad: bool
    motivos: list[str] = field(default_factory=list)
    latencia_ms: float = 0.0
    dominio: dict | None = None

    def como_dict(self) -> dict:
        return asdict(self)


class ModeloBaseline:
    clave = "baseline"

    def __init__(self, ruta: Path = cfg.RUTA_BASELINE, verificar: bool = True):
        import joblib

        p = cfg.parametros_baseline()
        if verificar and sha256_ruta(ruta) != p["sha256"]:
            raise ErrorModelo(f"La huella SHA-256 de {ruta} no coincide con registro_modelo.json")
        paquete = joblib.load(ruta)
        self.vectorizador = paquete["vectorizador"]
        self.modelo = paquete["modelo"]
        self.columna = paquete["columna"]
        self.clases = list(self.modelo.classes_)
        if self.clases != cfg.CLASES:
            raise ErrorModelo(f"Clases inesperadas en el baseline: {self.clases}")
        self.nombre = p["nombre"]
        self.umbral_abstencion = p["umbral_abstencion"]
        self.umbrales_clase = p["umbrales_clase"]
        self.clases_clave = cfg.clases_clave()
        n_ent = int(cfg.registro_modelo()["datos"]["entrenamiento"])
        self.filtro = FiltroDominio(self.vectorizador, n_ent)

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        return self.modelo.predict_proba(self.vectorizador.transform(preparar_baseline(textos, self.columna)))

    def decidir(self, prob: np.ndarray) -> np.ndarray:
        return aplicar_umbrales(prob, self.umbrales_clase, self.clases)

    def predecir(self, textos: list[str], filtros: list[ResultadoFiltro] | None = None) -> list[Prediccion]:
        t0 = time.perf_counter()
        prob = self.probabilidades(textos)
        emociones = self.decidir(prob)
        ms = (time.perf_counter() - t0) * 1000 / max(len(textos), 1)
        salida = []
        for i, fila in enumerate(prob):
            conf = float(fila.max())
            motivos = []
            if conf < self.umbral_abstencion:
                motivos.append(f"confianza baja ({conf:.2f} < {self.umbral_abstencion:.2f})")
            if filtros is not None and filtros[i].motivo and filtros[i].motivo.startswith("texto vacío"):
                motivos.append(filtros[i].motivo)
            salida.append(Prediccion(
                modelo=self.clave, emocion=str(emociones[i]), emocion_mas_probable=self.clases[int(fila.argmax())],
                confianza=round(conf, 4), probabilidades={c: round(float(p), 4) for c, p in zip(self.clases, fila)},
                revision_humana=bool(motivos), prioridad=str(emociones[i]) in self.clases_clave, motivos=motivos,
                latencia_ms=round(ms, 2), dominio=asdict(filtros[i]) if filtros is not None else None))
        return salida


class ModeloBert:
    clave = "bert"

    def __init__(self, ruta: Path = cfg.DIR_BERT, filtro: FiltroDominio | None = None, verificar: bool = True):
        if not (Path(ruta) / "model.safetensors").exists():
            raise ErrorModelo(f"BERT no está en {ruta}. Descárgalo con: python scripts/descargar_bert.py")
        p = cfg.parametros_bert()
        if verificar and sha256_ruta(ruta) != p["sha256"]:
            raise ErrorModelo(f"La huella SHA-256 de {ruta} no coincide con registro_modelo.json")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self._torch = torch
        self.tokenizador = AutoTokenizer.from_pretrained(ruta)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
        self.modelo.eval()
        self.clases = [self.modelo.config.id2label[i] for i in range(self.modelo.config.num_labels)]
        if self.clases != cfg.CLASES:
            raise ErrorModelo(f"Clases inesperadas en BERT: {self.clases}")
        self.nombre = p["nombre"]
        self.umbral_abstencion = p["umbral_abstencion"]
        self.max_len = p["max_len"]
        self.clases_clave = cfg.clases_clave()
        self.filtro = filtro

    def probabilidades(self, textos: list[str], bs: int = 64) -> np.ndarray:
        """Copia de `bert_probabilidades` (celda 80): ordena por longitud para reducir el relleno."""
        textos = preparar_bert(textos)
        orden = np.argsort([len(t) for t in textos])
        salida = np.zeros((len(textos), len(self.clases)))
        with self._torch.no_grad():
            for i in range(0, len(textos), bs):
                idx = orden[i:i + bs]
                b = self.tokenizador([textos[j] for j in idx], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt")
                salida[idx] = self._torch.softmax(self.modelo(**b).logits, dim=-1).numpy()
        return salida

    def predecir(self, textos: list[str], filtros: list[ResultadoFiltro] | None = None) -> list[Prediccion]:
        if filtros is None:
            filtros = [self.filtro.evaluar(t) for t in textos] if self.filtro else [None] * len(textos)
        salida: list[Prediccion | None] = [None] * len(textos)
        en_dominio = [i for i, f in enumerate(filtros) if f is None or f.en_dominio]
        for i, f in enumerate(filtros):
            if f is not None and not f.en_dominio:   # BERT no clasifica: va directo a revisión
                salida[i] = Prediccion(modelo=self.clave, emocion="sin clasificar", emocion_mas_probable="sin clasificar",
                                       confianza=0.0, probabilidades={}, revision_humana=True, prioridad=False,
                                       motivos=[f.motivo], latencia_ms=0.0, dominio=asdict(f))
        if en_dominio:
            t0 = time.perf_counter()
            prob = self.probabilidades([textos[i] for i in en_dominio])
            ms = (time.perf_counter() - t0) * 1000 / len(en_dominio)
            for fila, i in zip(prob, en_dominio):
                conf = float(fila.max())
                emocion = self.clases[int(fila.argmax())]
                motivos = [f"confianza baja ({conf:.2f} < {self.umbral_abstencion:.2f})"] if conf < self.umbral_abstencion else []
                salida[i] = Prediccion(
                    modelo=self.clave, emocion=emocion, emocion_mas_probable=emocion, confianza=round(conf, 4),
                    probabilidades={c: round(float(p), 4) for c, p in zip(self.clases, fila)},
                    revision_humana=bool(motivos), prioridad=emocion in self.clases_clave, motivos=motivos,
                    latencia_ms=round(ms, 2), dominio=asdict(filtros[i]) if filtros[i] is not None else None)
        return salida  # type: ignore[return-value]


class Servicio:
    """Mantiene los modelos cargados y une baseline y BERT en una sola respuesta."""

    def __init__(self, cargar_bert: bool = cfg.USAR_BERT):
        self.baseline = ModeloBaseline()
        self.bert: ModeloBert | None = None
        self.error_bert: str | None = None
        if cargar_bert:
            try:
                self.bert = ModeloBert(filtro=self.baseline.filtro)
            except ErrorModelo as e:
                self.error_bert = str(e)
        else:
            self.error_bert = "BERT desactivado (DELL_USAR_BERT=0)"

    def predecir(self, textos: list[str], modelo: str = "baseline") -> list[list[Prediccion]]:
        if modelo in ("bert", "ambos") and self.bert is None:
            raise ErrorModelo(self.error_bert or "BERT no está disponible")
        filtros = [self.baseline.filtro.evaluar(t) for t in textos]
        resultados: list[list[Prediccion]] = [[] for _ in textos]
        if modelo in ("baseline", "ambos"):
            for i, p in enumerate(self.baseline.predecir(textos, filtros)):
                resultados[i].append(p)
        if modelo in ("bert", "ambos"):
            for i, p in enumerate(self.bert.predecir(textos, filtros)):
                resultados[i].append(p)
        return resultados
