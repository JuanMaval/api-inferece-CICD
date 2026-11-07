# api-inference

Servicio FastAPI que expone el modelo de readmisión diabética registrado en
MLflow (`models:/diabetes@production`), registra cada inferencia en la tabla
`inference.logs` y publica métricas Prometheus.

Este repositorio contiene únicamente el servicio de inferencia y su pipeline
de pruebas. El entrenamiento, las migraciones de base de datos y el despliegue
viven en el repositorio `diabetes-mlops`; el contrato de datos compartido
(`diabetes_model`) está vendorizado aquí para que el servicio se construya y
se pruebe de forma autónoma.

## Estructura del repositorio

```text
.
├── src/api_inference/      # Aplicación FastAPI
│   ├── main.py             # create_app(), lifespan, rutas de operación
│   ├── predict.py          # POST /predict y /predict/{id}/feedback
│   ├── model.py            # Carga, refresco y recarga del modelo MLflow
│   ├── repository.py       # Persistencia en inference.logs / inference.feedback
│   ├── spool.py            # Spool en disco para filas que la base de datos no aceptó
│   ├── metrics.py          # Middleware Prometheus
│   ├── schemas.py          # Esquemas Pydantic generados desde el contrato
│   └── settings.py         # Configuración por variables de entorno
├── tests/                  # Suite pytest (sin red, MLflow ni PostgreSQL)
├── ml/diabetes_model/      # Contrato de datos compartido (copia vendorizada)
├── Dockerfile              # Imagen de producción (uvicorn)
├── Dockerfile.test         # Imagen que ejecuta la suite de pruebas
├── pyproject.toml          # Dependencias y configuración de pytest
└── uv.lock                 # Versiones fijadas (uv)
```

## Inicio rápido

Requisitos: [uv](https://docs.astral.sh/uv/) (descarga CPython 3.14 si no está
instalado) y, para las imágenes, Docker.

```bash
uv sync                 # crea .venv con dependencias y grupo dev
uv run pytest -q        # ejecuta la suite
```

Para ejecutar la suite en Docker, sin instalar nada en la máquina:

```bash
docker build -f Dockerfile.test -t api-inference-tests .
docker run --rm api-inference-tests
```

Más detalle en [Pruebas en Docker](#pruebas-en-docker).

Para levantar el servicio contra un MLflow y un PostgreSQL reales:

```bash
TRAIN_DATABASE_URL=postgresql://app:app@localhost:5432/diabetes \
MLFLOW_TRACKING_URI=http://localhost:5000 \
uv run uvicorn api_inference.main:app --reload
```

## Endpoints

| Método | Ruta            | Descripción |
| ------ | --------------- | ----------- |
| GET    | `/health`       | Liveness. Siempre `200`; informa `model_loaded` y el estado de la base de datos (`ok`/`error`). |
| GET    | `/model-info`   | Nombre, alias, versión, `run_id`, estado (`ready`/`unavailable`), `loaded_at`, última comprobación del alias y último error. |
| POST   | `/model/reload` | Fuerza la recarga de la versión apuntada por el alias. Devuelve `previous_version`, `current_version` y `changed`. `503` si la carga falla (se conserva la versión anterior). |
| POST   | `/predict`      | Una observación por petición. Contrato definido en `ml/diabetes_model/src/diabetes_model/contract.py`. |
| POST   | `/predict/{request_id}/feedback` | Registra el resultado real (`readmitted`) de una petición ya registrada. Alimenta `inference.training_examples`. |
| GET    | `/metrics`      | Exposición Prometheus (`text/plain; version=0.0.4`). |

### `POST /predict`

```json
{
  "request_id": "UUID opcional; se genera si falta",
  "features": { "race": "Caucasian", "time_in_hospital": 6, "...": "44 campos" }
}
```

- Los 44 campos del contrato son obligatorios (admiten `null`). Los ocho
  numéricos deben ser números JSON; el resto, cadenas (incluidos los códigos
  como `"admission_type_id": "1"`). Los tipos son estrictos: `"6"` en un campo
  numérico o `1` en un código se rechazan.
- `encounter_id`, `patient_nbr`, `readmitted`, `weight`, `examide`,
  `citoglipton` y cualquier clave desconocida producen `422`.
- El esquema se construye en tiempo de importación a partir de
  `diabetes_model.contract.INPUT_COLUMNS`/`NUMERIC_INPUTS`, de modo que la API
  y el pipeline de entrenamiento comparten la misma definición.

Respuesta `200`:

```json
{
  "request_id": "UUID",
  "prediction": "NO",
  "probabilities": { "NO": 0.8, ">30": 0.15, "<30": 0.05 },
  "model": { "name": "diabetes", "version": "7", "alias": "production" },
  "processing_time_ms": 4.2
}
```

Errores: `422` (`validation_error`), `503` (`model_unavailable`), `500`
(`inference_error`). El cuerpo incluye `request_id` y
`detail.error_code`/`detail.message`.

### `POST /predict/{request_id}/feedback`

Cierra el ciclo de reentrenamiento: asocia el resultado observado a una
petición ya registrada en `inference.logs`.

```json
{ "readmitted": "<30", "source": "ehr" }
```

- `readmitted` ∈ {`NO`, `>30`, `<30`}, el mismo dominio que
  `clean.records.target_readmitted`. `source` es texto libre opcional (máx.
  200 caracteres) con el origen de la etiqueta.
- Semántica *upsert*: una segunda llamada para el mismo `request_id`
  reemplaza la etiqueta (`created: false`). Así una corrección posterior gana.
- Se admite feedback para peticiones fallidas (`prediction: null`), de modo que
  una observación mal puntuada también puede entrar al dataset.

Respuesta `200`:

```json
{ "request_id": "UUID", "readmitted": "<30", "prediction": "NO", "created": true }
```

Errores: `404` (`unknown_request`, no hay fila en `inference.logs`), `422`
(etiqueta o cuerpo inválidos), `503` (`database_unavailable`: la escritura
falló; a diferencia del log de inferencia, el feedback **no** se encola, el
cliente debe reintentar).

## Estrategia de carga del modelo

El modelo se carga con `mlflow.sklearn.load_model` (formato `skops`) para
disponer de `predict_proba` y `classes_`. La estrategia combina tres
mecanismos:

1. **Carga en el arranque.** El `lifespan` resuelve el alias con
   `MlflowClient().get_model_version_by_alias` y carga la versión concreta
   (`models:/<name>/<version>`). Hay reintentos acotados
   (`MODEL_LOAD_RETRIES`, backoff lineal `MODEL_LOAD_BACKOFF_SECONDS`). Si se
   agotan, el proceso arranca igualmente en estado `unavailable`: `/health`
   responde `200` con `model_loaded=false` y `/predict` responde `503` mientras
   tanto. El contenedor no entra en bucle de reinicio.
2. **Refresco en segundo plano.** Una tarea `asyncio` consulta el alias cada
   `MODEL_REFRESH_SECONDS` (60 por defecto; `0` la desactiva). Si la versión
   cambia, se carga la nueva y se sustituye la referencia bajo un
   `asyncio.Lock`; las peticiones en vuelo terminan con la versión que
   tomaron. Una versión nueva promovida por el DAG entra en servicio sin
   redeploy ni reinicio.
3. **Recarga bajo demanda.** `POST /model/reload` fuerza la carga aunque la
   versión no haya cambiado (útil tras una promoción manual o para validar el
   despliegue).

Las llamadas a MLflow y a scikit-learn son bloqueantes; todas se ejecutan en
`asyncio.to_thread` para no detener el event loop.

## Registro en `inference.logs`

Se persiste una fila por petición a `/predict`, incluidas las fallidas, en la
base de datos de entrenamiento (esquema `inference`, creado por las
migraciones del repositorio `diabetes-mlops`):

- `input_data`: el objeto `features` tal como se envió (en errores de parseo
  JSON se guarda `{"raw": "<cuerpo truncado>"}`).
- `prediction` (cadena JSON), `probabilities`, `model_name`, `model_version`,
  `model_alias` y `processing_time_ms` en éxito.
- `status = failed` con `error_code` ∈ {`validation_error`,
  `model_unavailable`, `inference_error`} y `error_message` en fallo.

`processing_time_ms` mide normalización más inferencia; no incluye la
escritura en base de datos.

### Ground truth: `inference.feedback` y `inference.training_examples`

La migración `003_inference_feedback.sql` de `diabetes-mlops` añade la tabla
`inference.feedback` (una etiqueta por `request_id`, FK a `inference.logs`) y
la vista `inference.training_examples`, que une cada predicción exitosa con su
etiqueta observada (`features`, `readmitted`, `predicted_readmitted`, identidad
del modelo). Esa vista es la entrada para reentrenar con datos de producción:
sus columnas `features`/`readmitted` espejan `clean.records`
(`features`/`target_readmitted`).

**Decisión:** la escritura se espera dentro de la petición y la predicción se
devuelve igual aunque falle. Lo que cambia es qué pasa con la fila:

- **Fallo transitorio** (base de datos caída, timeout de conexión): la fila se
  anexa a un *spool* en disco (`INFERENCE_LOG_SPOOL_PATH`, JSONL, una fila por
  línea, `fsync` por escritura) y se incrementa `inference_log_spooled_total`.
  Una tarea en segundo plano reintenta cada `INFERENCE_LOG_REPLAY_SECONDS`
  (también al arrancar, así se recuperan las filas de un proceso anterior),
  reinsertando en el orden original y deteniéndose en el primer fallo
  transitorio sin perder el resto. `inference_log_spool_pending` expone la
  cola. El replay mueve el lote a `<spool>.replaying` y solo lo borra al
  terminar, así un apagado o caída del proceso a mitad de camino no pierde
  filas (las que ya habían entrado vuelven como `request_id` duplicado y se
  descartan). Cada fila lleva su `requested_at` original, no la hora del
  replay.
- **Fallo permanente** (`IntegrityError`/`DataError` de psycopg, p. ej.
  `request_id` duplicado o una fila ya insertada antes de perder la conexión):
  reintentar no sirve, la fila se descarta y cuenta en
  `inference_log_failures_total`.

Con el spool, "registrar cada solicitud" sobrevive a una caída de Postgres. El
límite es el disco del proceso: en Kubernetes el volumen es `emptyDir`, así que
sobrevive a la caída de la base de datos pero no a reprogramar el pod; en
compose es un volumen con nombre. El feedback (`/predict/{id}/feedback`) no se
encola: responde `503` y el cliente reintenta.

## Métricas

Middleware ASGI propio sobre `prometheus_client`, etiquetado por plantilla de
ruta (no por URL cruda) para mantener acotada la cardinalidad; las rutas no
reconocidas se agrupan bajo `path="unmatched"` y `/metrics` no se contabiliza.

| Métrica | Etiquetas | Significado |
| ------- | --------- | ----------- |
| `http_requests_total` | `method`, `path`, `status` | Peticiones HTTP. |
| `http_request_duration_seconds` | `method`, `path` | Histograma de latencia (buckets de 1 ms a 10 s). |
| `http_requests_in_progress` | — | Peticiones en curso. |
| `predictions_total` | `status` | Resultado de `/predict`: `succeeded`, `validation_error`, `model_unavailable`, `inference_error`. |
| `prediction_duration_seconds` | — | Latencia de normalización + inferencia. |
| `inference_log_failures_total` | — | Filas de `inference.logs` perdidas definitivamente (rechazadas por la base de datos o imposibles de encolar). |
| `inference_log_spooled_total` | — | Filas enviadas al spool en disco por base de datos no disponible. |
| `inference_log_replayed_total` | — | Filas del spool que luego llegaron a la base de datos. |
| `inference_log_spool_pending` | — | Filas esperando en el spool ahora mismo. |
| `model_info` | `name`, `version`, `alias` | Vale 1 para la versión servida; se reinicia en cada recarga. |
| `model_reloads_total` | `result` | Intentos de carga (`success`/`failure`). |

Además se exportan los colectores por defecto (`process_*` en Linux,
`python_info`).

## Configuración

La API lee su configuración del entorno mediante `pydantic-settings`
(`src/api_inference/settings.py`). Para servir el modelo real hacen falta al
menos `MLFLOW_TRACKING_URI` y la conexión a la base de datos; el resto toma su
valor por defecto.

| Variable | Por defecto | Descripción |
| -------- | ----------- | ----------- |
| `MLFLOW_TRACKING_URI` | — | Servidor MLflow (el cliente también la lee de forma nativa). |
| `MODEL_NAME` / `MODEL_ALIAS` | `diabetes` / `production` | Modelo registrado y alias servido. |
| `MODEL_REFRESH_SECONDS` | `60` | Periodo del refresco en segundo plano; `0` lo desactiva. |
| `MODEL_LOAD_RETRIES` / `MODEL_LOAD_BACKOFF_SECONDS` | `5` / `2` | Reintentos de la carga inicial. |
| `TRAIN_DATABASE_URL` | — | DSN completo; tiene prioridad sobre las partes. |
| `TRAIN_DB_HOST` / `TRAIN_DB_PORT` / `TRAIN_DB_NAME` / `TRAIN_DB_USER` / `TRAIN_DB_PASSWORD` | `—/5432/—/—/—` | Partes del DSN de `inference.logs`. |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `4` | Tamaño del pool async de psycopg. |
| `DB_CONNECT_TIMEOUT_SECONDS` | `5` | Timeout de conexión a la base de datos. |
| `INFERENCE_LOG_SPOOL_PATH` | `spool/inference-logs.jsonl` | Spool en disco para filas de `inference.logs` que la base de datos no aceptó (relativo al directorio de trabajo, `/app` en la imagen). Vacío lo desactiva. |
| `INFERENCE_LOG_REPLAY_SECONDS` | `15` | Periodo del reintento en segundo plano del spool; `0` lo desactiva. |
| `LOG_LEVEL` | `INFO` | Nivel de logging estándar. |

## Pruebas

Las pruebas no necesitan red, MLflow ni PostgreSQL: inyectan un registro falso,
un modelo falso y un repositorio en memoria a través de `create_app(...)`
(ver `tests/conftest.py`). La configuración de pytest (`testpaths`,
`asyncio_mode = "auto"`) vive en `pyproject.toml`.

```bash
uv run pytest -q                       # suite completa
uv run pytest tests/test_predict.py -q # un archivo
```

### Pruebas en Docker

`Dockerfile.test` construye una imagen con las dependencias fijadas en
`uv.lock` (incluido el grupo `dev`) y ejecuta la suite completa con pytest. El
contexto de construcción es la raíz del repositorio porque `uv.lock` resuelve
`diabetes-model` desde la copia vendorizada en `ml/diabetes_model`.

```bash
docker build -f Dockerfile.test -t api-inference-tests .
docker run --rm api-inference-tests                                  # suite completa
docker run --rm api-inference-tests pytest tests/test_predict.py -q  # un archivo
```

### CI y publicación en Docker Hub

El workflow `.github/workflows/ci-cd-unit-tests-dockerhub.yml` monitorea el
código (`src/`, `tests/`, `ml/`), los archivos de dependencias y los Dockerfiles
desde la raíz del repositorio (`APP_PATH: "."`). Ejecuta las pruebas en Python
3.14 para pushes a `main` y `feature/**`, y para pull requests hacia `main`.
Solo publica la imagen `api-inference` en Docker Hub después de pasar las
pruebas en un push a `main`.

Para habilitar la publicación, configura los secretos del repositorio
`DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`.

El contenedor termina con el código de salida de pytest, por lo que puede
usarse directamente como paso de CI.

## Imagen de producción

`Dockerfile` construye la imagen que sirve la API con uvicorn en el puerto
8000, con usuario no root, `HEALTHCHECK` sobre `/health` y el directorio
`/app/spool` preparado para el spool de `inference.logs`.

> **Pendiente:** el `Dockerfile` todavía usa las rutas del monorepo original
> (`apps/api-inference/...`). Hay que adaptarlo al layout de este repositorio
> (`src/`, `ml/diabetes_model`) antes de poder construirlo desde aquí.

## Dependencia `diabetes-model`

`ml/diabetes_model` es una copia del paquete `diabetes_model` de
`diabetes-mlops` (`ml/diabetes_model`). Define el contrato de entrada
(`INPUT_COLUMNS`, `NUMERIC_INPUTS`, `TARGET_CLASSES`, `input_frame`) a partir
del cual `schemas.py` genera los modelos Pydantic en tiempo de importación, de
modo que la API y el pipeline de entrenamiento validan exactamente las mismas
columnas. Se declara como dependencia de ruta en `pyproject.toml`
(`[tool.uv.sources]`) y está fijada en `uv.lock`.

Cuando el contrato cambie en `diabetes-mlops`, hay que copiar de nuevo
`pyproject.toml` y `src/diabetes_model/*.py` y regenerar el lock con
`uv lock`. Los artefactos `build/` y `*.egg-info/` que setuptools deja al
compilar el paquete están ignorados por git y por Docker.

## Versión de Python

El servicio se ejecuta sobre CPython 3.14 (`.python-version`). Se intentó
3.15, pero `pyarrow` (dependencia de MLflow) y `pandas 2.3.x` (fijado `<3` por
`diabetes_model`) no publican ruedas `cp315` y la compilación desde fuente de
`pyarrow` falla; 3.14 instala todo desde ruedas, incluido
`scikit-learn==1.9.1`, la misma versión con la que se entrena el modelo.
