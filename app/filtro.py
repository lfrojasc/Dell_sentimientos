"""Filtro de idioma y tema para BERT.

En las pruebas de robustez del notebook, BERT respondió con mucha confianza a textos que no
tienen nada que ver con Dell (por ejemplo, "The weather in Bogota is sunny today" -> joy 0,97).
Este filtro decide si un texto se parece a los tweets con los que se entrenó. Si no se parece,
BERT no lo clasifica y el tweet va a revisión humana.

Es una heurística sencilla y explicable, sin librerías nuevas:

1. **Idioma.** Cuenta palabras vacías (stop words) del inglés y de otros cinco idiomas, usando
   las listas de spaCy. Si hay al menos 2 de otro idioma y superan a las del inglés, el texto no
   está en inglés.
2. **Tema.** Si el texto menciona a Dell, pasa. Si no, al menos la mitad de sus palabras de
   contenido (las que no son stop words) deben aparecer en 10 o más tweets del entrenamiento.
   Eso se calcula con el vocabulario del TF-IDF del baseline, que ya trae esa información.
3. **Texto vacío.** Si después de limpiar no queda ninguna palabra (solo emojis o símbolos), no
   hay nada que clasificar.

Limitación: detecta otros idiomas, texto sin sentido y textos con vocabulario ajeno a Dell,
pero no todo texto fuera de tema escrito con palabras comunes (por ejemplo, "I love pizza").
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .limpieza import lematizar_uno, limpiar_texto

MIN_TWEETS_PALABRA = 10  # una palabra es "conocida" si aparece en al menos 10 tweets del entrenamiento
MIN_COBERTURA = 0.5  # fracción mínima de palabras de contenido conocidas
MIN_PALABRAS_OTRO_IDIOMA = 2
N_ENTRENAMIENTO = 19295  # tweets con los que se ajustó el vocabulario del baseline (registro_modelo.json)


@dataclass
class ResultadoFiltro:
    en_dominio: bool
    motivo: str | None
    idioma: str
    cobertura_vocabulario: float | None


@lru_cache(maxsize=1)
def _stop_words() -> tuple[frozenset, dict[str, frozenset]]:
    from spacy.lang.de.stop_words import STOP_WORDS as DE
    from spacy.lang.en.stop_words import STOP_WORDS as EN
    from spacy.lang.es.stop_words import STOP_WORDS as ES
    from spacy.lang.fr.stop_words import STOP_WORDS as FR
    from spacy.lang.it.stop_words import STOP_WORDS as IT
    from spacy.lang.pt.stop_words import STOP_WORDS as PT

    en = frozenset(EN)
    # Se quitan las palabras que también son del inglés ("a", "no", "me"...) para no confundir,
    # y "dell", que en italiano es una contracción ("dell'") y aquí es la marca.
    otros = {nombre: frozenset(sw) - en - {"dell"} for nombre, sw in
             {"español": ES, "portugués": PT, "francés": FR, "alemán": DE, "italiano": IT}.items()}
    return en, otros


class FiltroDominio:
    def __init__(self, vectorizador, n_entrenamiento: int = N_ENTRENAMIENTO):
        # Inverso de la fórmula del IDF suavizado de scikit-learn: idf = ln((1 + n) / (1 + df)) + 1
        n_tweets = (1 + n_entrenamiento) / np.exp(vectorizador.idf_ - 1) - 1
        self._conocidas = {w for w, j in vectorizador.vocabulary_.items()
                           if " " not in w and n_tweets[j] >= MIN_TWEETS_PALABRA - 0.5}

    def evaluar(self, texto: str) -> ResultadoFiltro:
        limpio = limpiar_texto(html.unescape(texto))
        palabras = limpio.split()
        if not palabras:
            return ResultadoFiltro(False, "texto vacío tras la limpieza (solo emojis o símbolos)", "desconocido", None)

        en, otros = _stop_words()
        n_en = sum(w in en for w in palabras)
        conteo = {nombre: sum(w in sw for w in palabras) for nombre, sw in otros.items()}
        idioma_otro, n_otro = max(conteo.items(), key=lambda kv: kv[1])
        if n_otro >= MIN_PALABRAS_OTRO_IDIOMA and n_otro > n_en:
            return ResultadoFiltro(False, f"fuera de dominio: el texto parece estar en {idioma_otro}", idioma_otro, None)
        idioma = "inglés" if n_en > 0 else "desconocido"

        contenido = [w for w in lematizar_uno(limpio).split() if w not in en]
        cobertura = (sum(w in self._conocidas for w in contenido) / len(contenido)) if contenido else 1.0
        if "dell" in limpio or "dell" in texto.lower():
            return ResultadoFiltro(True, None, idioma, round(cobertura, 3))
        if cobertura < MIN_COBERTURA:
            return ResultadoFiltro(False, f"fuera de dominio: solo el {cobertura:.0%} de las palabras se parecen "
                                          "a los tweets de Dell", idioma, round(cobertura, 3))
        return ResultadoFiltro(True, None, idioma, round(cobertura, 3))
