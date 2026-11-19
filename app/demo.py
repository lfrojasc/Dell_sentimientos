"""App de demostración (Gradio): escribir un tweet y ver la emoción según el baseline y BERT."""
from __future__ import annotations

import gradio as gr

from . import configuracion as cfg

ETIQUETAS = {
    "anger": "anger (ira)", "anticipation": "anticipation (expectativa)", "disgust": "disgust (disgusto)",
    "fear": "fear (miedo)", "joy": "joy (alegría)", "optimism": "optimism (optimismo)",
    "sadness": "sadness (tristeza)", "surprise": "surprise (sorpresa)",
}

EJEMPLOS = [
    ["@Dell my new laptop stopped working after two days and support never answers. Worst experience ever"],
    ["Thank you @Dell, the technician fixed my XPS in one visit. Great service!"],
    ["Waiting for my @Dell order, it ships next week. Can't wait to try it"],
    ["My Inspiron overheats every single day"],
    ["The weather in Bogota is sunny today"],
    ["El soporte técnico de Dell es pésimo y nadie responde"],
]


def _tarjeta(p) -> str:
    if p.emocion == "sin clasificar":
        encabezado = "### Sin clasificar → revisión humana"
    else:
        encabezado = f"### {ETIQUETAS.get(p.emocion, p.emocion)}"
    lineas = [encabezado]
    if p.emocion != "sin clasificar":
        lineas.append(f"**Confianza:** {p.confianza:.0%}")
        if p.emocion != p.emocion_mas_probable:
            lineas.append(f"*La más probable era {p.emocion_mas_probable}, pero el umbral por clase la cambió "
                          f"a {p.emocion} para no dejar pasar un cliente molesto.*")
    lineas.append("**Decisión:** " + ("👤 revisión humana" if p.revision_humana else "✅ el modelo decide solo"))
    if p.prioridad:
        lineas.append("**Prioridad:** 🔴 atender primero (anger o disgust)")
    for m in p.motivos:
        lineas.append(f"- {m}")
    if p.latencia_ms:
        lineas.append(f"<small>Tiempo: {p.latencia_ms:.1f} ms</small>")
    return "\n\n".join(lineas)


def _probabilidades(p) -> dict[str, float] | None:
    return {ETIQUETAS[c]: v for c, v in p.probabilidades.items()} if p.probabilidades else None


def construir_demo() -> gr.Blocks:
    from .api import clasificar, obtener_registro, obtener_servicio
    from .registro import resumen_monitoreo

    def predecir(texto: str):
        texto = (texto or "").strip()
        if not texto or len(texto) > cfg.LIMITE_CARACTERES:
            aviso = ("Escribe un tweet en el cuadro de arriba y presiona **Clasificar**." if not texto
                     else f"El texto tiene más de {cfg.LIMITE_CARACTERES} caracteres.")
            return aviso, None, "", None
        hay_bert = obtener_servicio().bert is not None
        preds = clasificar([texto], "ambos" if hay_bert else "baseline", origen="demo")[0]
        base = preds[0]
        if hay_bert:
            bert = preds[1]
            return _tarjeta(base), _probabilidades(base), _tarjeta(bert), _probabilidades(bert)
        aviso = "### BERT no está disponible\n\n" + (obtener_servicio().error_bert or "")
        return _tarjeta(base), _probabilidades(base), aviso, None

    def monitoreo(dias: float):
        datos = resumen_monitoreo(obtener_registro().leer(int(dias)))
        if not datos:
            return "Todavía no hay predicciones en este periodo."
        filas = ["| Indicador | " + " | ".join(datos) + " |", "|---|" + "---|" * len(datos)]
        nombres = [("predicciones", "Predicciones"), ("tasa_revision_humana", "Revisión humana"),
                   ("tasa_prioridad_anger_disgust", "Prioridad (anger/disgust)"),
                   ("tasa_fuera_de_dominio", "Fuera de dominio"), ("confianza_media", "Confianza media"),
                   ("latencia_p50_ms", "Latencia p50 (ms)"), ("latencia_p95_ms", "Latencia p95 (ms)"),
                   ("psi", "PSI vs referencia"), ("alerta", "Estado")]
        for clave, nombre in nombres:
            valores = []
            for d in datos.values():
                v = d.get(clave)
                if v is None:
                    valores.append("—")
                elif clave.startswith("tasa_"):
                    valores.append(f"{v:.0%}")
                else:
                    valores.append(str(v))
            filas.append(f"| {nombre} | " + " | ".join(valores) + " |")
        return "\n".join(filas)

    reg = cfg.registro_modelo()["metricas_prueba"]
    mb = next(v for k, v in reg.items() if k.startswith("Baseline"))
    mt = next(v for k, v in reg.items() if k.startswith("BERT"))
    pb, pt = cfg.parametros_baseline(), cfg.parametros_bert()

    with gr.Blocks(title="Emociones en tweets de Dell", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Emociones en tweets de clientes de Dell\n"
            "FLAJ S.A.S. · Proyecto final de Procesamiento de Lenguaje Natural.\n\n"
            "Escribe un tweet **en inglés** y compara el **modelo principal** (baseline TF-IDF + regresión logística) "
            "con el **modelo retador** (BERT). Los tweets con ira o disgusto se marcan para atender primero, y los "
            "dudosos van a revisión humana.")
        with gr.Tab("Clasificar"):
            texto = gr.Textbox(label="Tweet", lines=3, max_length=cfg.LIMITE_CARACTERES,
                               placeholder="@Dell my laptop arrived broken and nobody answers my emails")
            boton = gr.Button("Clasificar", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Modelo principal: baseline")
                    tarjeta_b = gr.Markdown()
                    prob_b = gr.Label(label="Probabilidades", num_top_classes=4)
                with gr.Column():
                    gr.Markdown("## Modelo retador: BERT")
                    tarjeta_t = gr.Markdown()
                    prob_t = gr.Label(label="Probabilidades", num_top_classes=4)
            gr.Examples(EJEMPLOS, inputs=texto)
            salidas = [tarjeta_b, prob_b, tarjeta_t, prob_t]
            boton.click(predecir, inputs=texto, outputs=salidas)
            texto.submit(predecir, inputs=texto, outputs=salidas)
        with gr.Tab("Monitoreo"):
            gr.Markdown("Indicadores de las predicciones registradas (no se guarda el texto de los tweets). "
                        "El PSI compara la mezcla de emociones con la de la partición de prueba: menos de 0,10 es "
                        "estable, de 0,10 a 0,25 hay que vigilar y más de 0,25 hay que investigar.")
            dias = gr.Slider(1, 90, value=7, step=1, label="Días hacia atrás")
            boton_m = gr.Button("Actualizar")
            tabla = gr.Markdown()
            boton_m.click(monitoreo, inputs=dias, outputs=tabla)
        with gr.Tab("Sobre los modelos"):
            gr.Markdown(
                "| | Baseline (principal) | BERT (retador) |\n|---|---|---|\n"
                f"| F1 clave (anger y disgust) | {mb['f1_clave']:.3f} | {mt['f1_clave']:.3f} |\n"
                f"| Exactitud | {mb['exactitud']:.1%} | {mt['exactitud']:.1%} |\n"
                f"| Latencia p95 por tweet | {mb['latencia_p95_ms']:.1f} ms | {mt['latencia_p95_ms']:.1f} ms |\n"
                f"| Umbral de revisión humana | confianza < {pb['umbral_abstencion']:.2f} | "
                f"confianza < {pt['umbral_abstencion']:.2f} o fuera de dominio |\n"
                f"| Umbral por clase | anger ≥ {pb['umbrales_clase']['anger']:.2f}, disgust ≥ "
                f"{pb['umbrales_clase']['disgust']:.2f} | — |\n\n"
                "Métricas en la partición de prueba (4.824 tweets). El baseline es el modelo principal porque BERT no "
                "cumple el límite de latencia del prerregistro (100 ms). BERT corre en paralelo como retador.\n\n"
                "**Limitaciones:** las etiquetas de entrenamiento las asignó otro modelo (no personas); solo inglés; "
                "tweets de 2022. El modelo apoya la priorización, no reemplaza la revisión humana.")
    return demo
