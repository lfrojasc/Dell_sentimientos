"""Registro de predicciones y monitoreo.

Cada predicción se guarda en una base SQLite (`registros/predicciones.sqlite`) con la fecha, el
modelo, la emoción, la confianza, si fue a revisión humana y la latencia.

**No se guarda el texto del tweet**, para no almacenar datos personales. Solo una huella
(SHA-256 recortada) que permite detectar repetidos sin poder leer el contenido.

El monitoreo compara la distribución de emociones predichas en un periodo con la distribución
predicha por el mismo modelo sobre la partición de prueba (`artefactos/referencia_monitoreo.json`).
Es decir, compara predicciones contra predicciones. Así el PSI mide el cambio en los datos y no
el sesgo propio del modelo.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import configuracion as cfg

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_utc TEXT NOT NULL,
    origen TEXT NOT NULL,
    modelo TEXT NOT NULL,
    emocion TEXT NOT NULL,
    emocion_mas_probable TEXT NOT NULL,
    confianza REAL NOT NULL,
    revision_humana INTEGER NOT NULL,
    prioridad INTEGER NOT NULL,
    fuera_de_dominio INTEGER NOT NULL,
    latencia_ms REAL NOT NULL,
    huella_texto TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fecha ON predicciones (fecha_utc);
"""


def psi(ref, act, eps: float = 1e-4) -> float:
    """Population Stability Index (misma fórmula que el notebook)."""
    ref, act = np.asarray(ref, dtype=float) + eps, np.asarray(act, dtype=float) + eps
    ref, act = ref / ref.sum(), act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def huella(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


class RegistroPredicciones:
    def __init__(self, ruta: Path = cfg.RUTA_LOG):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._candado = threading.Lock()
        with self._conexion() as con:
            con.executescript(_ESQUEMA)

    def _conexion(self) -> sqlite3.Connection:
        return sqlite3.connect(self.ruta, timeout=10)

    def guardar(self, textos: list[str], resultados: list[list], origen: str = "api") -> None:
        ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        filas = []
        for texto, preds in zip(textos, resultados):
            h = huella(texto)
            for p in preds:
                fuera = bool(p.dominio) and not p.dominio.get("en_dominio", True)
                filas.append((ahora, origen, p.modelo, p.emocion, p.emocion_mas_probable, p.confianza,
                              int(p.revision_humana), int(p.prioridad), int(fuera), p.latencia_ms, h))
        with self._candado, self._conexion() as con:
            con.executemany(
                "INSERT INTO predicciones (fecha_utc, origen, modelo, emocion, emocion_mas_probable, confianza, "
                "revision_humana, prioridad, fuera_de_dominio, latencia_ms, huella_texto) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", filas)

    def leer(self, dias: int | None = None) -> list[dict]:
        consulta, args = "SELECT * FROM predicciones", ()
        if dias is not None:
            desde = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat(timespec="seconds")
            consulta, args = consulta + " WHERE fecha_utc >= ?", (desde,)
        with self._conexion() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(consulta + " ORDER BY id", args)]


def _referencia() -> dict | None:
    if not cfg.RUTA_REFERENCIA_MONITOREO.exists():
        return None
    with open(cfg.RUTA_REFERENCIA_MONITOREO, encoding="utf-8") as fh:
        return json.load(fh)


def resumen_monitoreo(filas: list[dict]) -> dict:
    """Indicadores por modelo: volumen, distribución, PSI, revisión humana, confianza y latencia."""
    ref = _referencia()
    salida = {}
    for modelo in sorted({f["modelo"] for f in filas}):
        g = [f for f in filas if f["modelo"] == modelo]
        clasificados = [f for f in g if f["emocion"] in cfg.CLASES]
        conteo = np.array([sum(f["emocion"] == c for f in clasificados) for c in cfg.CLASES], dtype=float)
        dist = conteo / conteo.sum() if conteo.sum() else conteo
        lat = np.array([f["latencia_ms"] for f in g if f["latencia_ms"] > 0])
        ind = {
            "predicciones": len(g),
            "distribucion_emociones": {c: round(float(p), 4) for c, p in zip(cfg.CLASES, dist)},
            "tasa_revision_humana": round(float(np.mean([f["revision_humana"] for f in g])), 4),
            "tasa_prioridad_anger_disgust": round(float(np.mean([f["prioridad"] for f in g])), 4),
            "tasa_fuera_de_dominio": round(float(np.mean([f["fuera_de_dominio"] for f in g])), 4),
            "confianza_media": round(float(np.mean([f["confianza"] for f in clasificados])), 4) if clasificados else None,
            "latencia_p50_ms": round(float(np.percentile(lat, 50)), 2) if len(lat) else None,
            "latencia_p95_ms": round(float(np.percentile(lat, 95)), 2) if len(lat) else None,
            "psi": None,
            "alerta": None,
        }
        if ref and modelo in ref.get("modelos", {}):
            r = ref["modelos"][modelo]
            ind["referencia"] = {k: r[k] for k in ("distribucion_emociones", "tasa_revision_humana", "confianza_media")}
            if len(clasificados) >= cfg.MIN_TWEETS_PSI:
                valor = psi([r["distribucion_emociones"][c] for c in cfg.CLASES], dist)
                ind["psi"] = round(valor, 4)
                ind["alerta"] = ("estable" if valor < cfg.PSI_VIGILAR else
                                 "cambio moderado: vigilar" if valor < cfg.PSI_INVESTIGAR else
                                 "cambio fuerte: investigar o reentrenar")
            else:
                ind["alerta"] = f"pocos datos: el PSI se calcula desde {cfg.MIN_TWEETS_PSI} predicciones"
        salida[modelo] = ind
    return salida
