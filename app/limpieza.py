"""Limpieza del texto: copia exacta de las funciones del notebook.

Regla de oro: el modelo debe recibir en producción el mismo texto que vio en el entrenamiento.
Por eso estas funciones NO se "mejoran" aquí. Cualquier cambio debe hacerse primero en el
notebook, reentrenar los modelos y luego copiarse.

Detalle conocido (documentado en el informe): la expresión de menciones `@[A-Za-z0-9_]+` no
reconoce letras con tilde, así que una mención como `@Ďellcares` no se elimina por completo.
Se mantiene igual porque los modelos se entrenaron así.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache


def texto_para_bert(texto: str) -> str:
    """Limpieza mínima para BERT (celda 30 del notebook)."""
    texto = re.sub(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\S*", "[email]", texto)  # correos -> marcador
    texto = re.sub(r"@[A-Za-z0-9_]+", "@user", texto)  # menciones -> marcador
    texto = re.sub(r"https?://\S+|www\.\S+", "http", texto)  # enlaces -> marcador
    return re.sub(r"\s+", " ", texto).strip()


def limpiar_texto(texto: str) -> str:
    """Normalización para el TF-IDF y las redes (celda 32 del notebook)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()  # minúsculas
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)  # menciones
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)  # enlaces
    texto = re.sub(r"[^\w\s]", "", texto)  # puntuación y símbolos
    return re.sub(r"\s+", " ", texto).strip()  # espacios


@lru_cache(maxsize=1)
def _nlp_lemas():
    """Lematizador de spaCy en modo lookup (celda 38 del notebook)."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
    nlp.initialize()
    return nlp


def lematizar_uno(texto: str) -> str:
    return " ".join(tok.lemma_.lower() for tok in _nlp_lemas()(texto) if not tok.is_space)


def lematizar_lote(textos: list[str]) -> list[str]:
    """Igual que `lematizar_uno`, pero más rápido para muchos textos (como hace el notebook con `pipe`)."""
    return [
        " ".join(tok.lemma_.lower() for tok in doc if not tok.is_space)
        for doc in _nlp_lemas().pipe(textos, batch_size=1000)
    ]


def preparar_baseline(textos: list[str], columna: str = "Texto lematizado") -> list[str]:
    """Texto crudo -> entrada del baseline (función `prep_base` de la celda 86 del notebook)."""
    limpios = [limpiar_texto(html.unescape(t)) for t in textos]
    return lematizar_lote(limpios) if columna == "Texto lematizado" else limpios


def preparar_bert(textos: list[str]) -> list[str]:
    """Texto crudo -> entrada de BERT (función `prob_bert_crudo` de la celda 86 del notebook)."""
    return [texto_para_bert(html.unescape(t)) for t in textos]
