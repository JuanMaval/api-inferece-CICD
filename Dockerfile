# Inference API image.
#
# Dependencies are installed from pyproject.toml + uv.lock, so the runtime
# resolves exactly the versions pinned in the lockfile.
#
# Build context is the repository root: uv.lock resolves the shared
# diabetes-model package from the vendored ml/diabetes_model directory, so the
# deps stage mirrors the repository layout under /src and places the virtualenv
# at /app/.venv via UV_PROJECT_ENVIRONMENT:
#   docker build -t api-inference .
#   docker run --rm -p 8000:8000 \
#     -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#     -e TRAIN_DATABASE_URL=postgresql://app:app@postgres:5432/diabetes \
#     api-inference

# --------------------------------------------------------------------------- #
# Dependencies (venv built with uv, no project install)
# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /src
# Path dependency resolved by uv.lock as ml/diabetes_model.
COPY ml/diabetes_model ./ml/diabetes_model
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
FROM python:3.14-slim-trixie AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY src/ /app/src/

# Mount point of the inference.logs spool (INFERENCE_LOG_SPOOL_PATH defaults
# to spool/inference-logs.jsonl under /app). It must exist in the image, owned
# by the runtime user: a named volume mounted on a missing path is created
# root-owned and the API could not write to it.
RUN mkdir -p /app/spool && chown -R 1000:1000 /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api_inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
