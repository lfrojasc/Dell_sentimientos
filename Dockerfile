# Imagen del servicio: API (FastAPI) + demo (Gradio).
#   Con BERT:     docker build -t dell-emociones .
#   Sin BERT:     docker build --build-arg INCLUIR_BERT=0 -t dell-emociones .
#   Ejecutar:     docker run -p 8000:8000 dell-emociones   ->  http://localhost:8000
FROM python:3.12-slim

ARG INCLUIR_BERT=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    DELL_USAR_BERT=${INCLUIR_BERT}

WORKDIR /servicio

COPY requirements.txt requirements-bert.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$INCLUIR_BERT" = "1" ]; then pip install --no-cache-dir -r requirements-bert.txt; fi

COPY app ./app
COPY scripts ./scripts
COPY artefactos/baseline.joblib artefactos/registro_modelo.json artefactos/referencia_monitoreo.json ./artefactos/

# BERT se descarga de la Release de GitHub y se verifica su huella SHA-256.
RUN if [ "$INCLUIR_BERT" = "1" ]; then python scripts/descargar_bert.py; fi

RUN useradd --create-home servicio && mkdir -p registros && chown -R servicio /servicio
USER servicio

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
