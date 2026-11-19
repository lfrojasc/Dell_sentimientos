"""Reproduce la evaluación en prueba del notebook con el código del despliegue.

Qué hace:
1. Rehace la limpieza y la partición del notebook a partir del CSV (misma semilla, mismos pasos).
2. Clasifica la partición de prueba con los modelos desplegados.
3. Compara las métricas con las que registró el notebook (`registro_modelo.json`).
4. Guarda la distribución de predicciones de referencia para el monitoreo
   (`artefactos/referencia_monitoreo.json`).

Uso:
    python scripts/reproducir_prueba.py            # baseline y BERT (BERT tarda unos 10 a 15 min en CPU)
    python scripts/reproducir_prueba.py --sin-bert # solo el baseline (segundos)
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import configuracion as cfg  # noqa: E402
from app.limpieza import lematizar_lote, limpiar_texto, texto_para_bert  # noqa: E402

# Las probabilidades de BERT (float32) suman 1 con un error de ~1e-7; scikit-learn lo avisa sin que importe.
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")

SEMILLA = 42
TOLERANCIA = 1e-4   # el registro guarda las métricas redondeadas a 4 decimales


def preparar_datos(ruta_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pasos 1 a 8 de la limpieza y la partición (celdas 26 a 46 del notebook)."""
    df = pd.read_csv(ruta_csv)
    d = df.drop_duplicates(subset=["Text", "Username"], keep="first").copy()
    d["Text decodificado"] = d["Text"].map(html.unescape)
    d["Texto BERT"] = d["Text decodificado"].map(texto_para_bert)
    d["Texto limpio"] = d["Text decodificado"].map(limpiar_texto)
    d = d.drop_duplicates(subset=["Texto limpio", "Username"], keep="first").copy()
    d["Texto base"] = (d["Texto limpio"].str.replace(r"\d+", "", regex=True)
                       .str.replace(r"\s+", " ", regex=True).str.strip())
    d["Texto lematizado"] = lematizar_lote(d["Texto limpio"].tolist())
    d = d[d["Texto limpio"].str.len() > 0].copy()
    d = d.reset_index(drop=False).rename(columns={"index": "fila_original"})
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    idx_tr, idx_te = next(sgkf.split(d, d["emotion"], groups=d["Texto base"]))
    return d.iloc[idx_tr], d.iloc[idx_te]


def metricas(y_real, y_pred, prob) -> dict:
    """Copia de `calcular_metricas` (celda 50 del notebook)."""
    clave = cfg.clases_clave()
    f1 = f1_score(y_real, y_pred, labels=clave, average=None, zero_division=0)
    rc = recall_score(y_real, y_pred, labels=clave, average=None, zero_division=0)
    return {"f1_clave": f1.mean(), "f1_macro": f1_score(y_real, y_pred, average="macro", zero_division=0),
            "f1_anger": f1[0], "f1_disgust": f1[1], "recall_anger": rc[0], "recall_disgust": rc[1],
            "exactitud": float(np.mean(np.asarray(y_real) == np.asarray(y_pred))),
            "log_loss": log_loss(y_real, prob, labels=cfg.CLASES)}


def comparar(nombre: str, obtenidas: dict, registro: dict) -> bool:
    ok = True
    print(f"\n{nombre}")
    print(f"  {'métrica':<15}{'notebook':>10}{'despliegue':>12}")
    for k, v in obtenidas.items():
        esperado = registro[k]
        coincide = abs(round(v, 4) - esperado) <= TOLERANCIA
        ok &= coincide
        print(f"  {k:<15}{esperado:>10.4f}{v:>12.4f}  {'ok' if coincide else 'DIFERENTE'}")
    return ok


def referencia(prob: np.ndarray, emociones: np.ndarray, umbral: float) -> dict:
    conteo = pd.Series(emociones).value_counts(normalize=True).reindex(cfg.CLASES).fillna(0)
    return {"distribucion_emociones": {c: round(float(p), 6) for c, p in conteo.items()},
            "tasa_revision_humana": round(float((prob.max(axis=1) < umbral).mean()), 6),
            "confianza_media": round(float(prob.max(axis=1).mean()), 6),
            "tweets": int(len(emociones))}


def principal(sin_bert: bool, ruta_csv: Path, guardar: bool) -> int:
    reg = cfg.registro_modelo()
    huella_csv = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    if huella_csv != reg["datos"]["sha256"]:
        print(f"El CSV no es el mismo que usó el notebook (huella {huella_csv[:16]}...)")
        return 2
    t = time.perf_counter()
    df_tr, df_te = preparar_datos(ruta_csv)
    print(f"Partición: entrenamiento {len(df_tr)} | prueba {len(df_te)} ({time.perf_counter() - t:.0f} s)")
    if (len(df_tr), len(df_te)) != (reg["datos"]["entrenamiento"], reg["datos"]["prueba"]):
        print("El tamaño de la partición no coincide con el registro del notebook")
        return 1
    y_te = df_te["emotion"].values
    todo_ok = True
    refs = {}

    from app.modelos import ModeloBaseline, ModeloBert
    base = ModeloBaseline()
    crudos = df_te["Text"].tolist()
    prob_b = base.probabilidades(crudos)   # desde el texto crudo, como en producción
    prob_nb = base.modelo.predict_proba(base.vectorizador.transform(df_te["Texto lematizado"]))
    print(f"Diferencia máxima entre el camino de producción y el del notebook: {np.abs(prob_b - prob_nb).max():.2e}")
    pred_b = np.array(cfg.CLASES)[prob_b.argmax(axis=1)]
    todo_ok &= comparar(base.nombre, metricas(y_te, pred_b, prob_b), reg["metricas_prueba"][base.nombre])
    refs["baseline"] = referencia(prob_b, base.decidir(prob_b), base.umbral_abstencion)

    if not sin_bert:
        bert = ModeloBert()
        t = time.perf_counter()
        prob_t = bert.probabilidades(crudos)
        print(f"\nBERT clasificó {len(crudos)} tweets en {(time.perf_counter() - t) / 60:.1f} min")
        pred_t = np.array(cfg.CLASES)[prob_t.argmax(axis=1)]
        todo_ok &= comparar(bert.nombre, metricas(y_te, pred_t, prob_t), reg["metricas_prueba"][bert.nombre])
        filtros = [base.filtro.evaluar(x) for x in crudos]
        fuera = np.array([not f.en_dominio for f in filtros])
        print(f"Filtro de idioma y tema: {fuera.sum()} de {len(fuera)} tweets de prueba quedan fuera de dominio "
              f"({fuera.mean():.2%})")
        refs["bert"] = referencia(prob_t, pred_t, bert.umbral_abstencion)
        refs["bert"]["tasa_fuera_de_dominio"] = round(float(fuera.mean()), 6)

    if guardar:
        anterior = {}
        if cfg.RUTA_REFERENCIA_MONITOREO.exists():
            anterior = json.loads(cfg.RUTA_REFERENCIA_MONITOREO.read_text(encoding="utf-8")).get("modelos", {})
        anterior.update(refs)
        salida = {"descripcion": "Predicciones de cada modelo desplegado sobre la partición de prueba del notebook. "
                                 "Es la referencia del PSI en el monitoreo (predicciones contra predicciones).",
                  "datos_sha256": huella_csv, "modelos": anterior}
        cfg.RUTA_REFERENCIA_MONITOREO.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReferencia de monitoreo guardada en {cfg.RUTA_REFERENCIA_MONITOREO}")

    print("\nRESULTADO:", "las métricas coinciden con el notebook" if todo_ok else "HAY DIFERENCIAS")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-bert", action="store_true", help="evaluar solo el baseline")
    ap.add_argument("--csv", type=Path, default=cfg.RUTA_DATOS)
    ap.add_argument("--no-guardar", action="store_true", help="no actualizar referencia_monitoreo.json")
    a = ap.parse_args()
    sys.exit(principal(a.sin_bert, a.csv, not a.no_guardar))
