"""Descarga BERT (438 MB) de la Release de GitHub y verifica su huella SHA-256.

BERT no cabe en el repositorio (GitHub no acepta archivos de más de 100 MB), así que se publica
como archivo adjunto de una Release. Este script lo descarga, lo descomprime en
`artefactos/bert_final/` y comprueba que la huella coincide con la que registró el notebook.

Uso:
    python scripts/descargar_bert.py
    python scripts/descargar_bert.py --url <otra URL>   # si la Release tiene otro nombre
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import configuracion as cfg  # noqa: E402
from app.modelos import sha256_ruta  # noqa: E402


def descargar(url: str, destino: Path) -> None:
    print(f"Descargando {url}")
    with urllib.request.urlopen(url) as r, open(destino, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        leido, siguiente = 0, 0.1
        while bloque := r.read(1 << 20):
            fh.write(bloque)
            leido += len(bloque)
            if total and leido / total >= siguiente:
                print(f"  {leido / total:.0%} ({leido / 1e6:,.0f} de {total / 1e6:,.0f} MB)", flush=True)
                siguiente += 0.1


def principal(url: str) -> int:
    esperado = cfg.parametros_bert()["sha256"]
    if cfg.DIR_BERT.exists() and sha256_ruta(cfg.DIR_BERT) == esperado:
        print(f"BERT ya está en {cfg.DIR_BERT} y su huella es correcta.")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        zip_local = Path(tmp) / "bert_final.zip"
        descargar(url, zip_local)
        extraido = Path(tmp) / "extraido"
        with zipfile.ZipFile(zip_local) as z:
            z.extractall(extraido)
        carpeta = extraido / "bert_final"
        obtenida = sha256_ruta(carpeta)
        if obtenida != esperado:
            print(f"ERROR: la huella no coincide.\n  esperada: {esperado}\n  obtenida: {obtenida}")
            return 1
        if cfg.DIR_BERT.exists():
            shutil.rmtree(cfg.DIR_BERT)
        cfg.DIR_BERT.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(carpeta), str(cfg.DIR_BERT))
    print(f"BERT listo en {cfg.DIR_BERT} (huella verificada).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=cfg.URL_BERT)
    sys.exit(principal(ap.parse_args().url))
