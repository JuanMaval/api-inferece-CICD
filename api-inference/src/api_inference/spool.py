"""Durable buffer for ``inference.logs`` rows the database could not take.

``DurableInferenceLog`` writes a row to the repository and, when the database
is unreachable, appends it to an on-disk JSONL spool instead of dropping it.
``replay`` re-inserts spooled rows in their original order once the database
answers again; the lifespan runs it at startup and periodically.

Two files make the replay crash-safe: ``<spool>`` receives new rows and
``<spool>.replaying`` holds the batch a replay is working on. A replay moves
the pending rows into the in-flight file, inserts them, then writes whatever
is left back in front of ``<spool>`` and removes the in-flight file. A crash
or cancellation at any point leaves every row in one of the two files; the
next replay picks the in-flight file up first. A row that was inserted just
before the crash comes back as a duplicate ``request_id`` and is dropped.

Only transient failures are spooled. A row the database rejects for good
(integrity or data errors, e.g. a duplicate ``request_id``) would fail on
every replay, so it is dropped and counted in ``inference_log_failures_total``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg

from api_inference.metrics import (
    INFERENCE_LOG_FAILURES,
    INFERENCE_LOG_REPLAYED,
    INFERENCE_LOG_SPOOL_PENDING,
    INFERENCE_LOG_SPOOLED,
)
from api_inference.repository import InferenceLogRecord, InferenceLogRepository

logger = logging.getLogger(__name__)

_RECORD_FIELDS = {field.name for field in fields(InferenceLogRecord)}
IN_FLIGHT_SUFFIX = ".replaying"


def is_permanent_failure(exc: BaseException) -> bool:
    """True when retrying the same row can never succeed."""
    return isinstance(exc, psycopg.errors.IntegrityError | psycopg.errors.DataError)


def encode_record(record: InferenceLogRecord) -> str:
    payload = asdict(record)
    payload["request_id"] = str(record.request_id)
    if record.requested_at is not None:
        payload["requested_at"] = record.requested_at.isoformat()
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def decode_record(line: str) -> InferenceLogRecord:
    payload: dict[str, Any] = json.loads(line)
    if not isinstance(payload, dict) or set(payload) - _RECORD_FIELDS:
        raise ValueError("not an inference log record")
    payload["request_id"] = UUID(payload["request_id"])
    if payload.get("requested_at") is not None:
        payload["requested_at"] = datetime.fromisoformat(payload["requested_at"])
    return InferenceLogRecord(**payload)


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class InferenceLogSpool:
    """Append-only JSONL files, one ``InferenceLogRecord`` per line.

    All methods are synchronous and run in a worker thread by the caller;
    ``DurableInferenceLog`` serialises file access with a lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.in_flight_path = path.with_name(path.name + IN_FLIGHT_SUFFIX)

    def append(self, record: InferenceLogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(encode_record(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def pending(self) -> int:
        """Rows waiting in either file (in-flight rows are still pending)."""
        return len(self._lines(self.in_flight_path)) + len(self._lines(self.path))

    def read_all(self) -> list[InferenceLogRecord]:
        """Every pending row, oldest first: an unfinished batch, then new rows."""
        return self._decode(self._lines(self.in_flight_path) + self._lines(self.path))

    def take(self) -> list[InferenceLogRecord]:
        """Move every pending row into the in-flight file and return them.

        The rows stay on disk until ``settle`` runs, so a crash or a cancelled
        replay loses nothing: the next ``take`` reads the in-flight file first.
        """
        lines = self._lines(self.in_flight_path) + self._lines(self.path)
        if not lines:
            self.in_flight_path.unlink(missing_ok=True)
            return []
        _write_atomic(self.in_flight_path, lines)
        self.path.unlink(missing_ok=True)
        return self._decode(lines)

    def settle(self, remaining: list[InferenceLogRecord]) -> None:
        """Close a replay: put ``remaining`` back in front of the rows appended
        meanwhile and drop the in-flight file."""
        lines = [encode_record(record) for record in remaining] + self._lines(self.path)
        if lines:
            _write_atomic(self.path, lines)
        else:
            self.path.unlink(missing_ok=True)
        self.in_flight_path.unlink(missing_ok=True)

    def _decode(self, lines: list[str]) -> list[InferenceLogRecord]:
        records: list[InferenceLogRecord] = []
        for number, line in enumerate(lines, start=1):
            try:
                records.append(decode_record(line))
            except (ValueError, TypeError, KeyError):
                logger.error("dropping unreadable spool line %d in %s", number, self.path)
                INFERENCE_LOG_FAILURES.inc()
        return records

    @staticmethod
    def _lines(path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return [line for line in text.splitlines() if line.strip()]


class DurableInferenceLog:
    """Repository writes with an on-disk fallback and ordered replay."""

    def __init__(
        self, repository: InferenceLogRepository, spool: InferenceLogSpool | None
    ) -> None:
        self._repository = repository
        self._spool = spool
        self._lock = asyncio.Lock()

    @property
    def spool_path(self) -> Path | None:
        return self._spool.path if self._spool else None

    async def write(self, record: InferenceLogRecord) -> bool:
        """Persist the row now, or spool it; never raises. True when stored."""
        try:
            await self._repository.log(record)
            return True
        except Exception as exc:
            if self._spool is None or is_permanent_failure(exc):
                INFERENCE_LOG_FAILURES.inc()
                logger.exception("inference log write failed for request %s", record.request_id)
                return False
            logger.warning(
                "inference log write failed for request %s, spooling: %s",
                record.request_id, exc,
            )
        try:
            async with self._lock:
                await asyncio.to_thread(self._spool.append, record)
                INFERENCE_LOG_SPOOL_PENDING.set(await asyncio.to_thread(self._spool.pending))
        except Exception:
            INFERENCE_LOG_FAILURES.inc()
            logger.exception("inference log spool append failed for request %s", record.request_id)
            return False
        INFERENCE_LOG_SPOOLED.inc()
        return False

    async def replay(self) -> int:
        """Re-insert spooled rows in order; stop at the first transient failure.

        Returns how many rows reached the database. Rows the database rejects
        for good are dropped so one bad row cannot block the queue. Whatever
        the outcome (including cancellation), the rows not confirmed stored
        are written back to the spool before this returns.
        """
        if self._spool is None:
            return 0
        async with self._lock:
            records = await asyncio.to_thread(self._spool.take)
            if not records:
                INFERENCE_LOG_SPOOL_PENDING.set(0)
                return 0

        replayed = 0
        remaining = records
        try:
            for index, record in enumerate(records):
                # The row being inserted stays in ``remaining`` until the insert
                # is confirmed: if we are cancelled mid-flight it is replayed
                # again and, when it did land, dropped as a duplicate.
                remaining = records[index:]
                try:
                    await self._repository.log(record)
                    replayed += 1
                except Exception as exc:
                    if is_permanent_failure(exc):
                        INFERENCE_LOG_FAILURES.inc()
                        logger.warning(
                            "dropping spooled inference log %s rejected by the database: %s",
                            record.request_id, exc,
                        )
                        continue
                    logger.warning("inference log replay paused, database unavailable: %s", exc)
                    break
            else:
                remaining = []
        finally:
            # Shielded so a task cancellation cannot skip the write-back.
            await asyncio.shield(self._settle(remaining))
        if replayed:
            INFERENCE_LOG_REPLAYED.inc(replayed)
            logger.info("replayed %d spooled inference log rows", replayed)
        return replayed

    async def _settle(self, remaining: list[InferenceLogRecord]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._spool.settle, remaining)
            INFERENCE_LOG_SPOOL_PENDING.set(await asyncio.to_thread(self._spool.pending))
