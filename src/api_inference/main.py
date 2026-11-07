"""FastAPI application factory for the diabetes inference service."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.exceptions import RequestValidationError

from api_inference.metrics import METRICS_PATH, MetricsMiddleware, metrics_response
from api_inference.model import ModelLoadError, ModelService
from api_inference.predict import build_router, build_validation_handler
from api_inference.repository import InferenceLogRepository, PostgresInferenceLogRepository
from api_inference.schemas import ErrorDetail, HealthResponse, ModelInfoResponse, ReloadResponse
from api_inference.settings import Settings, get_settings
from api_inference.spool import DurableInferenceLog, InferenceLogSpool

logger = logging.getLogger("api_inference")

API_VERSION = "0.1.0"


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level.upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    root.setLevel(level.upper())


async def _initial_load(service: ModelService, settings: Settings) -> None:
    """Bounded startup retries; on exhaustion the API starts ``unavailable``."""
    for attempt in range(1, settings.model_load_retries + 1):
        try:
            await service.load()
            return
        except ModelLoadError as exc:
            logger.warning(
                "initial model load attempt %d/%d failed: %s",
                attempt, settings.model_load_retries, exc,
            )
            if attempt < settings.model_load_retries:
                await asyncio.sleep(settings.model_load_backoff_seconds * attempt)
    logger.error(
        "model %s@%s unavailable after %d attempts; serving 503 until the "
        "background refresh succeeds",
        settings.model_name, settings.model_alias, settings.model_load_retries,
    )


async def _refresh_loop(service: ModelService, interval: float) -> None:
    """Poll the registry alias forever; ``refresh_if_changed`` never raises."""
    while True:
        await asyncio.sleep(interval)
        try:
            if service.loaded is None:
                await service.load()
            else:
                await service.refresh_if_changed()
        except ModelLoadError:
            pass  # already recorded in ``service.last_error``
        except Exception:  # keep the loop alive on unexpected errors
            logger.exception("model refresh loop error")


async def _replay_loop(inference_log: DurableInferenceLog, interval: float) -> None:
    """Replay spooled ``inference.logs`` rows at startup and then periodically."""
    while True:
        try:
            await inference_log.replay()
        except Exception:  # keep the loop alive on unexpected errors
            logger.exception("inference log replay loop error")
        await asyncio.sleep(interval)


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
    repository: InferenceLogRepository | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)
    if settings.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)

    service = model_service or ModelService(
        name=settings.model_name,
        alias=settings.model_alias,
        refresh_seconds=settings.model_refresh_seconds,
    )
    repo = repository or PostgresInferenceLogRepository(
        lambda: settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        connect_timeout_seconds=settings.db_connect_timeout_seconds,
    )
    spool = (
        InferenceLogSpool(Path(settings.inference_log_spool_path))
        if settings.inference_log_spool_path
        else None
    )
    inference_log = DurableInferenceLog(repo, spool)

    async def _cancel(task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await repo.open()
        refresh_task: asyncio.Task[None] | None = None
        replay_task: asyncio.Task[None] | None = None
        try:
            if spool is not None and settings.inference_log_replay_seconds > 0:
                replay_task = asyncio.create_task(
                    _replay_loop(inference_log, settings.inference_log_replay_seconds),
                    name="inference-log-replay",
                )
            await _initial_load(service, settings)
            if settings.model_refresh_seconds > 0:
                refresh_task = asyncio.create_task(
                    _refresh_loop(service, settings.model_refresh_seconds),
                    name="model-refresh",
                )
            yield
        finally:
            await _cancel(refresh_task)
            await _cancel(replay_task)
            await repo.close()

    app = FastAPI(title="Diabetes Inference API", version=API_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.model_service = service
    app.state.repository = repo
    app.state.inference_log = inference_log
    app.include_router(build_router(service, repo, inference_log))
    app.add_exception_handler(RequestValidationError, build_validation_handler(inference_log))
    app.add_middleware(MetricsMiddleware)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness probe: 200 even without a model so the container stays up."""
        try:
            await repo.ping()
            database = "ok"
        except Exception as exc:
            logger.warning("health: database ping failed: %s", exc)
            database = "error"
        return HealthResponse(
            status="ok", model_loaded=service.loaded is not None, database=database
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    async def model_info() -> ModelInfoResponse:
        info = asdict(service.info())
        info["classes"] = list(info["classes"]) if info["classes"] else None
        return ModelInfoResponse(**info, tracking_uri=settings.mlflow_tracking_uri)

    @app.post("/model/reload", response_model=ReloadResponse, tags=["model"])
    async def reload_model() -> ReloadResponse:
        """Force a reload of whatever version the alias points to right now."""
        previous = service.loaded.version if service.loaded else None
        try:
            loaded = await service.load()
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ErrorDetail(error_code="model_load_error", message=str(exc)).model_dump(),
            ) from exc
        return ReloadResponse(
            previous_version=previous,
            current_version=loaded.version,
            changed=previous != loaded.version,
        )

    @app.get(METRICS_PATH, include_in_schema=False)
    async def metrics() -> Response:
        return metrics_response()

    return app


app = create_app()
