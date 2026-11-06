# Inference API (S4) image.
#
# Dependencies are installed from apps/api-inference/pyproject.toml + uv.lock,
# so the runtime resolves exactly the versions pinned in the lockfile.
#
# Build context is the repository root because the API depends on the shared
# ml/diabetes_model package (uv.lock records it as ../../ml/diabetes_model
# relative to the project), so the deps stage mirrors the repository layout
# under /src and places the virtualenv at /app/.venv via UV_PROJECT_ENVIRONMENT:
#   docker compose build api-inference
#   docker build -f apps/api-inference/Dockerfile -t api-inference .

# --------------------------------------------------------------------------- #
# Dependencies (venv built with uv, no project install)
# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Path dependency resolved by uv.lock as ../../ml/diabetes_model.
COPY ml/diabetes_model /src/ml/diabetes_model

WORKDIR /src/apps/api-inference
COPY apps/api-inference/pyproject.toml apps/api-inference/uv.lock ./

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
COPY apps/api-inference/src/ /app/src/

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
