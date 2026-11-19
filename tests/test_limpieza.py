from app.limpieza import limpiar_texto, preparar_baseline, preparar_bert, texto_para_bert


def test_limpiar_quita_menciones_enlaces_y_puntuacion():
    assert limpiar_texto("@Dell My LAPTOP is broken!!! https://t.co/abc") == "my laptop is broken"


def test_limpiar_texto_no_valido():
    assert limpiar_texto(None) == ""


def test_html_se_decodifica_antes_de_limpiar():
    # El notebook corrigió que "&amp;" se volviera la palabra "amp".
    assert preparar_baseline(["Dell &amp; HP"], columna="Texto limpio") == ["dell hp"]


def test_baseline_usa_lemas():
    assert preparar_baseline(["My laptops were broken"]) == ["my laptop be break"]


def test_bert_enmascara_datos_personales():
    texto = "@Dell write to ana.perez@mail.com or see https://dell.com/x"
    assert texto_para_bert(texto) == "@user write to [email] or see http"


def test_bert_conserva_mayusculas_y_puntuacion():
    assert preparar_bert(["WORST laptop &amp; support!!"]) == ["WORST laptop & support!!"]


def test_mencion_con_tilde_igual_que_el_notebook():
    # Detalle documentado: la regex de menciones no reconoce letras con tilde. Se conserva a propósito.
    assert limpiar_texto("@Ďellcares help") == "ďellcares help"
