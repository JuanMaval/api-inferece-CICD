"""Prometheus instrumentation: pure-ASGI HTTP middleware plus domain metrics.

Everything registers on the default ``REGISTRY`` so the process/platform
collectors (CPU, RSS, open fds, Python version) are exported alongside.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

METRICS_PATH = "/metrics"
# Label used for requests that matched no route (404/405): a constant keeps
# the series cardinality bounded even under random path scanning.
UNMATCHED_PATH = "unmatched"

# Millisecond-level latency buckets: the model answers in a few ms, the slow
# tail (cold thread, DB hiccup) sits in the hundreds of ms.
LATENCY_BUCKETS = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests by method, route template and status code.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency by method and route template.",
    ["method", "path"],
    buckets=LATENCY_BUCKETS,
)
HTTP_IN_PROGRESS = Gauge(
    "http_requests_in_progress", "HTTP requests currently being handled."
)
PREDICTIONS = Counter(
    "predictions_total",
    "Prediction requests by outcome "
    "(succeeded, validation_error, model_unavailable, inference_error).",
    ["status"],
)
PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Normalisation plus model inference latency for successful predictions.",
    buckets=LATENCY_BUCKETS,
)
INFERENCE_LOG_FAILURES = Counter(
    "inference_log_failures_total",
    "inference.logs rows lost for good: rejected by the database or unspoolable "
    "(prediction still served).",
)
INFERENCE_LOG_SPOOLED = Counter(
    "inference_log_spooled_total",
    "inference.logs rows written to the on-disk spool because the database was unavailable.",
)
INFERENCE_LOG_REPLAYED = Counter(
    "inference_log_replayed_total",
    "Spooled inference.logs rows that later reached the database.",
)
INFERENCE_LOG_SPOOL_PENDING = Gauge(
    "inference_log_spool_pending",
    "inference.logs rows currently waiting in the on-disk spool.",
)
MODEL_INFO = Gauge(
    "model_info",
    "Currently served model; always 1 for the active name/version/alias.",
    ["name", "version", "alias"],
)
MODEL_RELOADS = Counter(
    "model_reloads_total", "Model load attempts by result (success, failure).", ["result"]
)


def set_model_info(name: str, version: str, alias: str) -> None:
    """Point ``model_info`` at the new version, dropping the previous series."""
    MODEL_INFO.clear()
    MODEL_INFO.labels(name=name, version=version, alias=alias).set(1)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """Pure ASGI middleware: no body buffering, labels by route template.

    The route is read from ``scope["route"]`` after the inner app ran, which
    FastAPI sets when a route matches; unmatched requests get a constant label.
    ``/metrics`` itself is not measured.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == METRICS_PATH:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        status_code = 500
        started = time.perf_counter()
        HTTP_IN_PROGRESS.inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            HTTP_IN_PROGRESS.dec()
            route = scope.get("route")
            path = getattr(route, "path", None) or UNMATCHED_PATH
            HTTP_REQUESTS.labels(method=method, path=path, status=str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(
                time.perf_counter() - started
            )
