# Emociones en tweets de clientes de Dell

[![Pruebas](https://github.com/lfrojasc/Dell_sentimientos/actions/workflows/pruebas.yml/badge.svg)](https://github.com/lfrojasc/Dell_sentimientos/actions/workflows/pruebas.yml)

**FLAJ S.A.S.** · Proyecto final de Procesamiento de Lenguaje Natural

**Equipo:** Jessica Cuenu Fuentes, Laura Marcela Garnica Tapia, Oscar Andres Ramos Villanueva y Luisa Fernanda Rojas Chavarro.

El proyecto clasifica tweets que mencionan a Dell en 8 emociones, para que el equipo de atención priorice a los clientes con **ira (anger)** o **disgusto (disgust)**. Los casos dudosos se envían a revisión humana.

Este repositorio contiene:
- el notebook;
- el informe técnico;
- los datos;
- el **despliegue**: una API, una app de demostración, Docker y pruebas automáticas.

## Resultados

Medidos en la partición de prueba: 4.824 tweets que el modelo nunca vio.

| Modelo | F1 clave (anger y disgust) | Exactitud | Latencia p95 por tweet |
|---|---|---|---|
| **Baseline TF-IDF de lemas + regresión logística** (modelo principal) | 0,570 | 60,0 % | 4,3 ms |
| BiLSTM | 0,572 | 55,7 % | 9,2 ms |
| **BERT bert-base-uncased** (modelo retador) | **0,707** | **74,4 %** | 189,5 ms |

**Decisión.** BERT es claramente mejor en calidad. Pero la regla fijada antes de ver los resultados (el prerregistro) exigía una latencia p95 de 100 ms o menos, y BERT no la cumple en CPU. Por eso se despliega:
- el **baseline como modelo principal**;
- **BERT como retador**, que corre en paralelo para compararlo (*shadow mode*).

El detalle está en el informe ([docs/](docs/)).

## Contenido

```
├── app/                    código del despliegue
│   ├── limpieza.py         limpieza del texto (copia exacta del notebook)
│   ├── modelos.py          carga de modelos, verificación de huellas y política de decisión
│   ├── filtro.py           filtro de idioma y tema para BERT
│   ├── registro.py         registro de predicciones y monitoreo (PSI)
│   ├── api.py              API REST (FastAPI)
│   └── demo.py             app de demostración (Gradio)
├── artefactos/             modelos entrenados y registro del notebook
├── datos/                  dataset original (CSV)
├── docs/                   informe técnico (Word y PDF)
├── notebooks/              notebook ejecutado
├── scripts/                descargar BERT y reproducir la evaluación
├── tests/                  pruebas automáticas
├── Dockerfile
└── .github/workflows/      pruebas automáticas en GitHub Actions
```

## Datos

- **Fuente:** [Sentiment & Emotions Labelled Tweets](https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets) en Kaggle.
- **Contenido:** 24.970 menciones públicas a @Dell, de enero a septiembre de 2022.
- **Licencia:** CDLA-Permissive-1.0. Los tweets están sujetos a los términos de X/Twitter.
- **Etiquetas:** las asignó un modelo de Transformers, no personas (etiquetas "plata").
- **Datos personales:** el CSV contiene nombres de usuario y texto público tal como se publicó. Los modelos no usan el nombre de usuario, y el servicio desplegado no guarda el texto de los tweets.

---

# Cómo usar el despliegue, paso a paso

Los comandos son para **PowerShell en Windows**. Cada bloque se copia y se pega en la terminal.

## 1. Lo que necesitas instalado

- **Python 3.12.** Para comprobarlo, ejecuta `python --version`.
- **Git.** Para comprobarlo, ejecuta `git --version`.

Docker **no** es necesario para usarlo en tu computador.

## 2. Descargar el proyecto e instalarlo (solo la primera vez)

```powershell
git clone https://github.com/lfrojasc/Dell_sentimientos.git
cd Dell_sentimientos
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-bert.txt -r requirements-dev.txt
```

Esto crea un entorno aislado (`.venv`) con las mismas versiones de librerías que usó el notebook. La instalación tarda unos minutos, porque PyTorch es pesado.

Si solo quieres el baseline, omite `-r requirements-bert.txt`. La instalación es más rápida y la demo muestra un aviso en la columna de BERT.

## 3. Descargar BERT (solo la primera vez)

```powershell
.venv\Scripts\python.exe scripts\descargar_bert.py
```

El script descarga 406 MB desde la [Release v1.0](https://github.com/lfrojasc/Dell_sentimientos/releases) del repositorio y comprueba que el modelo sea exactamente el que se entrenó en el notebook (huella SHA-256).

## 4. Arrancar el servicio

```powershell
.venv\Scripts\python.exe -m uvicorn app.api:app --port 8000
```

Espera a que aparezca `Application startup complete`. Con BERT tarda unos 20 segundos. **Deja esa ventana abierta** mientras uses el servicio. Para apagarlo, presiona `Ctrl + C`.

## 5. Usar la app de demostración

Abre en el navegador **http://localhost:8000**. La app tiene tres pestañas:

- **Clasificar.** Escribe un tweet en inglés y presiona *Clasificar*. Verás dos columnas, el modelo principal (baseline) y el retador (BERT). Cada una muestra:
  - la **emoción** y la **confianza**;
  - la **decisión**: ✅ si el modelo decide solo, o 👤 si el tweet va a revisión humana, con el motivo;
  - la **prioridad**: 🔴 si es ira o disgusto (atender primero);
  - las probabilidades de las emociones más probables.

  Abajo hay ejemplos listos para probar, incluidos textos que no son de Dell. Con esos, BERT responde "sin clasificar".
- **Monitoreo.** Indicadores de todo lo que se ha clasificado (ver la sección 7).
- **Sobre los modelos.** Las métricas y los umbrales de cada modelo.

## 6. Usar la API

La API es la forma en que otro sistema (por ejemplo, la herramienta de atención de Dell) usaría el modelo.

**Opción fácil: desde el navegador.** Abre **http://localhost:8000/docs**. Ahí está cada servicio documentado:
1. Elige `POST /predecir`.
2. Presiona *Try it out*.
3. Cambia el texto de ejemplo.
4. Presiona *Execute*.

**Desde PowerShell:**

```powershell
$cuerpo = @{ texto = "@Dell my laptop arrived broken and nobody answers"; modelo = "ambos" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predecir -Method Post -ContentType "application/json" -Body $cuerpo | ConvertTo-Json -Depth 5
```

`modelo` puede ser `baseline` (por defecto), `bert` o `ambos`.

**Respuesta (resumida):**

```json
{
  "resultados": [
    {
      "modelo": "baseline",
      "emocion": "anger",
      "emocion_mas_probable": "anger",
      "confianza": 0.62,
      "probabilidades": {"anger": 0.62, "disgust": 0.15, "...": 0.0},
      "revision_humana": false,
      "prioridad": true,
      "motivos": [],
      "latencia_ms": 3.1
    }
  ]
}
```

| Servicio | Para qué sirve |
|---|---|
| `POST /predecir` | Clasifica un tweet |
| `POST /predecir_lote` | Clasifica hasta 100 tweets en una sola llamada (`{"textos": [...], "modelo": "baseline"}`) |
| `GET /monitoreo?dias=7` | Indicadores de los últimos días |
| `GET /modelo` | Umbrales, métricas de prueba, versiones y huellas de los modelos |
| `GET /salud` | Comprueba que el servicio está arriba y si BERT está cargado |

## 7. Leer el monitoreo

Cada predicción queda registrada en `registros/predicciones.sqlite`. Se guardan la fecha, el modelo, la emoción, la confianza, la decisión y la latencia. **No se guarda el texto del tweet**, solo una huella que no permite leerlo.

El monitoreo (pestaña *Monitoreo* o `GET /monitoreo`) muestra por modelo:

| Indicador | Qué significa |
|---|---|
| Revisión humana | Porcentaje de tweets enviados a una persona. En la prueba fue ~40 % para el baseline |
| Prioridad | Porcentaje clasificado como ira o disgusto |
| Fuera de dominio | Textos que no parecen tweets de Dell. Se mide para ambos modelos, pero solo BERT los deja sin clasificar |
| Confianza media | Si baja con el tiempo, los datos están cambiando |
| Latencia p50 / p95 | Tiempo por tweet |
| **PSI** | Cuánto cambió la mezcla de emociones frente a la de la partición de prueba: < 0,10 estable; 0,10–0,25 vigilar; > 0,25 investigar o reentrenar. Se calcula desde 30 predicciones |

## 8. Correr las pruebas

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Las pruebas comprueban:
- la limpieza;
- los umbrales;
- el filtro de idioma y tema;
- la API;
- que el registro no guarda el texto;
- que **el despliegue da las mismas predicciones que el notebook**: la tabla de robustez y las métricas del baseline sobre los 4.824 tweets de prueba.

La verificación completa de BERT sobre la partición de prueba tarda unos 10–15 minutos:

```powershell
.venv\Scripts\python.exe scripts\reproducir_prueba.py
```

En GitHub, las pruebas corren solas en cada cambio (pestaña *Actions*). También se construye la imagen de Docker para comprobar que funciona. Esa construcción usa solo el baseline, para no descargar BERT cada vez.

## 9. Docker (opcional)

Si tienes [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado:

```powershell
docker build -t dell-emociones .
docker run -p 8000:8000 dell-emociones
```

Luego abre http://localhost:8000. Para una imagen más liviana, solo con el baseline, agrega `--build-arg INCLUIR_BERT=0` al `docker build`.

---

# Cómo decide el sistema

**Baseline (modelo principal):**
1. Calcula la probabilidad de cada emoción.
2. Aplica el **umbral por clase**: si anger ≥ 0,25 o disgust ≥ 0,20, gana esa emoción aunque no sea la más probable. Así no se deja pasar un cliente molesto.
3. Aplica la **abstención**: si la emoción más probable tiene menos de 0,35 de probabilidad, el tweet va a **revisión humana**.

**BERT (modelo retador):**
1. Pasa por el **filtro de idioma y tema**. En las pruebas del notebook, BERT respondía con mucha confianza a textos ajenos a Dell. Por ejemplo, "The weather in Bogota is sunny today" daba joy con 0,97. Si el texto está en otro idioma, no tiene palabras, o casi no comparte vocabulario con los tweets de Dell, BERT no lo clasifica y el tweet va a revisión humana.
2. Si pasa el filtro, se toma la emoción más probable, con abstención cuando la confianza es menor de 0,45.

Los umbrales no están escritos a mano en el código: se leen de `artefactos/registro_modelo.json`, el registro que generó el notebook.

**Controles del despliegue:**
- Cada modelo se verifica con su **huella SHA-256** antes de cargarse. Si el archivo cambió, el servicio no lo usa.
- La limpieza es una **copia exacta** de la del notebook. Hay un detalle conocido: una mención con tilde, como `@Ďellcares`, no se elimina por completo. Se mantiene así porque los modelos se entrenaron de esa forma.

**Limitaciones:**
- El filtro es una heurística. Detecta otros idiomas, texto sin sentido y vocabulario ajeno, pero no todo texto fuera de tema escrito con palabras comunes.
- Las etiquetas de entrenamiento son de otro modelo, y los datos son solo en inglés y de 2022.
- El sistema apoya la priorización; no reemplaza la revisión humana.

## Uso de IA generativa

El equipo usó asistentes de IA como apoyo, según la declaración del Anexo F del informe. El equipo revisó el código y los resultados, y es responsable de su contenido.
