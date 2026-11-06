"""Persistence of one ``inference.logs`` row per request."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceLogRecord:
    """Mirror of the ``inference.logs`` columns written by the API."""

    request_id: UUID
    input_data: dict[str, Any]
    status: str  # "succeeded" | "failed"
    # Request time captured by the API; a row replayed from the spool after an
    # outage must keep it instead of taking the column default (insert time).
    requested_at: datetime | None = None
    prediction: str | None = None
    probabilities: dict[str, float] | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    processing_time_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class LoggedRequest:
    """The subset of an ``inference.logs`` row that feedback needs to see."""

    request_id: UUID
    status: str
    prediction: str | None


@dataclass(frozen=True)
class FeedbackRecord:
    """Mirror of the ``inference.feedback`` columns written by the API."""

    request_id: UUID
    readmitted: str  # one of diabetes_model.contract.TARGET_CLASSES
    source: str | None = None


class InferenceLogRepository(Protocol):
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> None: ...

    async def log(self, record: InferenceLogRecord) -> None: ...

    async def get_log(self, request_id: UUID) -> LoggedRequest | None: ...

    async def upsert_feedback(self, record: FeedbackRecord) -> bool:
        """Insert or replace the label; ``True`` when the row was created."""
        ...


_INSERT = """
INSERT INTO inference.logs (
    request_id, requested_at, input_data, prediction, probabilities,
    model_name, model_version, model_alias, processing_time_ms,
    status, error_code, error_message
) VALUES (
    %(request_id)s, COALESCE(%(requested_at)s, CURRENT_TIMESTAMP),
    %(input_data)s, %(prediction)s, %(probabilities)s,
    %(model_name)s, %(model_version)s, %(model_alias)s, %(processing_time_ms)s,
    %(status)s, %(error_code)s, %(error_message)s
)
"""

_SELECT_LOG = """
SELECT request_id, status, prediction #>> '{}'
FROM inference.logs
WHERE request_id = %(request_id)s
"""

# ``xmax = 0`` is true only for a freshly inserted row, so one round trip
# tells whether the label was created or replaced.
_UPSERT_FEEDBACK = """
INSERT INTO inference.feedback (request_id, readmitted, source, recorded_at)
VALUES (%(request_id)s, %(readmitted)s, %(source)s, CURRENT_TIMESTAMP)
ON CONFLICT (request_id) DO UPDATE SET
    readmitted = EXCLUDED.readmitted,
    source = EXCLUDED.source,
    recorded_at = EXCLUDED.recorded_at
RETURNING (xmax = 0) AS created
"""


class PostgresInferenceLogRepository:
    """Async psycopg3 pool against the training database (``postgres-train``)."""

    def __init__(
        self,
        dsn: Callable[[], str],
        *,
        min_size: int = 1,
        max_size: int = 4,
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        # The DSN is resolved lazily so importing the app never requires the
        # database environment; a missing configuration fails at startup.
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = connect_timeout_seconds
        self._pool: AsyncConnectionPool | None = None

    async def open(self) -> None:
        self._pool = AsyncConnectionPool(
            self._dsn(),
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            timeout=self._timeout,
            kwargs={"connect_timeout": int(self._timeout)},
        )
        # Do not wait for the first connection: a slow database must not keep
        # the API from starting. ``/health`` reports the real state.
        await self._pool.open(wait=False)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("repository is not open")
        return self._pool

    async def ping(self) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute("SELECT 1")

    async def log(self, record: InferenceLogRecord) -> None:
        params = {
            "request_id": record.request_id,
            "requested_at": record.requested_at,
            "input_data": Jsonb(record.input_data),
            # ``prediction`` is a JSONB column: store the label as a JSON string.
            "prediction": Jsonb(record.prediction) if record.prediction is not None else None,
            "probabilities": Jsonb(record.probabilities)
            if record.probabilities is not None
            else None,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "model_alias": record.model_alias,
            "processing_time_ms": record.processing_time_ms,
            "status": record.status,
            "error_code": record.error_code,
            "error_message": record.error_message,
        }
        async with self._require_pool().connection() as conn:
            await conn.execute(_INSERT, params)

    async def get_log(self, request_id: UUID) -> LoggedRequest | None:
        async with self._require_pool().connection() as conn:
            cursor = await conn.execute(_SELECT_LOG, {"request_id": request_id})
            row = await cursor.fetchone()
        if row is None:
            return None
        return LoggedRequest(request_id=row[0], status=row[1], prediction=row[2])

    async def upsert_feedback(self, record: FeedbackRecord) -> bool:
        params = {
            "request_id": record.request_id,
            "readmitted": record.readmitted,
            "source": record.source,
        }
        async with self._require_pool().connection() as conn:
            cursor = await conn.execute(_UPSERT_FEEDBACK, params)
            row = await cursor.fetchone()
        return bool(row and row[0])
