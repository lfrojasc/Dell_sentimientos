import pytest


@pytest.mark.parametrize("texto", [
    "My new Dell laptop works perfectly, I love it",
    "HP sent me a broken monitor and refuses to replace it",
    "My Inspiron overheats every single day",
    "this is so bad",
    "@Dell service is awesome #dell",   # "dell" también es palabra vacía del italiano: no debe confundir
])
def test_tweets_de_dominio_pasan(servicio, texto):
    assert servicio.baseline.filtro.evaluar(texto).en_dominio


@pytest.mark.parametrize("texto, motivo", [
    ("The weather in Bogota is sunny today", "palabras"),
    ("asdf qwerty zxcv", "palabras"),
    ("El soporte técnico de Dell es pésimo y nadie responde", "español"),
    ("O suporte da Dell é péssimo e ninguém responde", "portugués"),
    ("😡😡😡", "texto vacío"),
])
def test_textos_fuera_de_dominio(servicio, texto, motivo):
    r = servicio.baseline.filtro.evaluar(texto)
    assert not r.en_dominio and motivo in r.motivo
