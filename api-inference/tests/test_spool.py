"""Durable spool: inference.logs rows survive a database outage and are replayed."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import psycopg

from conftest import InMemoryRepository, make_app, valid_features
from api_inference.repository import InferenceLogRecord
from api_inference.spool import DurableInferenceLog, InferenceLogSpool, encode_record
from test_metrics import sample_value, scrape


def _record(**overrides) -> InferenceLogRecord:
    base = dict(
        request_id=uuid.uuid4(),
        input_data={"race": "Caucasian", "time_in_hospital": 6},
        status="succeeded",
        prediction="NO",
        probabilities={"NO": 0.8, ">30": 0.15, "<30": 0.05},
        model_name="diabetes",
        model_version="3",
        model_alias="production",
        processing_time_ms=1.25,
    )
    base.update(overrides)
    return InferenceLogRecord(**base)


# --------------------------------------------------------------------------- #
# DurableInferenceLog unit behaviour
# --------------------------------------------------------------------------- #
async def test_write_goes_to_repository_when_database_is_up(tmp_path: Path):
    repository = InMemoryRepository()
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(repository, spool)

    assert await durable.write(_record()) is True

    assert len(repository.rows) == 1
    assert spool.pending() == 0
    assert not (tmp_path / "spool.jsonl").exists()


async def test_transient_failure_spools_and_replay_recovers_in_order(tmp_path: Path):
    repository = InMemoryRepository(fail_log=True)
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(repository, spool)
    first, second = _record(), _record(status="failed", prediction=None, error_code="x")

    assert await durable.write(first) is False
    assert await durable.write(second) is False
    assert repository.rows == []
    assert spool.pending() == 2

    repository.fail_log = False
    assert await durable.replay() == 2

    assert repository.rows == [first, second]  # exact round trip, original order
    assert spool.pending() == 0


async def test_replay_stops_at_first_transient_failure_and_keeps_the_rest(tmp_path: Path):
    repository = InMemoryRepository(fail_log=True)
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(repository, spool)
    records = [_record() for _ in range(3)]
    for record in records:
        await durable.write(record)

    assert await durable.replay() == 0  # still down: nothing lost, nothing reordered
    assert spool.pending() == 3
    assert spool.read_all() == records


async def test_permanent_failure_is_dropped_not_spooled(tmp_path: Path):
    class DuplicateRepository(InMemoryRepository):
        async def log(self, record: InferenceLogRecord) -> None:
            raise psycopg.errors.UniqueViolation("duplicate key value")

    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(DuplicateRepository(), spool)

    assert await durable.write(_record()) is False

    assert spool.pending() == 0


async def test_replay_drops_rows_the_database_rejects_for_good(tmp_path: Path):
    """A row already stored before the connection dropped is a duplicate on replay."""
    repository = InMemoryRepository(fail_log=True)
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(repository, spool)
    duplicate, fresh = _record(), _record()
    await durable.write(duplicate)
    await durable.write(fresh)

    class RecoveredRepository(InMemoryRepository):
        async def log(self, record: InferenceLogRecord) -> None:
            if record.request_id == duplicate.request_id:
                raise psycopg.errors.UniqueViolation("duplicate key value")
            self.rows.append(record)

    recovered = RecoveredRepository()
    durable = DurableInferenceLog(recovered, spool)

    assert await durable.replay() == 1

    assert recovered.rows == [fresh]
    assert spool.pending() == 0


async def test_spool_survives_a_restart(tmp_path: Path):
    """A new process reading the same file replays what the old one spooled."""
    path = tmp_path / "spool.jsonl"
    record = _record()
    await DurableInferenceLog(InMemoryRepository(fail_log=True), InferenceLogSpool(path)).write(
        record
    )

    repository = InMemoryRepository()
    assert await DurableInferenceLog(repository, InferenceLogSpool(path)).replay() == 1
    assert repository.rows == [record]


async def test_cancelled_replay_keeps_every_unconfirmed_row(tmp_path: Path):
    """A shutdown in the middle of a replay (SIGTERM during an outage) loses nothing."""

    class SlowRepository(InMemoryRepository):
        async def log(self, record: InferenceLogRecord) -> None:
            await asyncio.sleep(10)  # a hanging connection attempt

    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(InMemoryRepository(fail_log=True), spool)
    records = [_record() for _ in range(3)]
    for record in records:
        await durable.write(record)

    task = asyncio.create_task(DurableInferenceLog(SlowRepository(), spool).replay())
    await asyncio.sleep(0.05)  # inside the first insert
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert spool.pending() == 3
    assert spool.read_all() == records
    repository = InMemoryRepository()
    assert await DurableInferenceLog(repository, spool).replay() == 3
    assert repository.rows == records


async def test_rows_appended_during_a_replay_are_kept_after_it(tmp_path: Path):
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    durable = DurableInferenceLog(InMemoryRepository(fail_log=True), spool)
    older = _record()
    await durable.write(older)
    newer = _record()

    class AppendingRepository(InMemoryRepository):
        async def log(self, record: InferenceLogRecord) -> None:
            if not self.rows:
                await durable.write(newer)  # a request arrives mid-replay
            raise RuntimeError("still down")

    await DurableInferenceLog(AppendingRepository(), spool).replay()

    assert spool.read_all() == [older, newer]
    assert not spool.in_flight_path.exists()


async def test_batch_left_in_flight_by_a_crash_is_replayed_first(tmp_path: Path):
    spool = InferenceLogSpool(tmp_path / "spool.jsonl")
    crashed, fresh = _record(), _record()
    spool.in_flight_path.write_text(encode_record(crashed) + "\n", encoding="utf-8")
    spool.append(fresh)
    assert spool.pending() == 2

    repository = InMemoryRepository()
    assert await DurableInferenceLog(repository, spool).replay() == 2

    assert repository.rows == [crashed, fresh]
    assert not spool.in_flight_path.exists()
    assert not spool.path.exists()


def test_requested_at_survives_the_spool_round_trip(tmp_path: Path):
    from datetime import UTC, datetime

    from api_inference.spool import decode_record

    record = _record(requested_at=datetime(2026, 11, 1, 22, 30, 15, 123456, tzinfo=UTC))

    assert decode_record(encode_record(record)) == record


async def test_corrupt_spool_line_is_skipped_with_the_rest_replayed(tmp_path: Path):
    path = tmp_path / "spool.jsonl"
    record = _record()
    spool = InferenceLogSpool(path)
    spool.append(record)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    spool.append(_record(request_id=uuid.UUID(int=1)))

    repository = InMemoryRepository()
    assert await DurableInferenceLog(repository, spool).replay() == 2
    assert [row.request_id for row in repository.rows] == [record.request_id, uuid.UUID(int=1)]
    assert spool.pending() == 0


def test_spool_file_is_one_json_object_per_line(tmp_path: Path):
    path = tmp_path / "nested" / "spool.jsonl"  # parent directories are created
    record = _record()
    InferenceLogSpool(path).append(record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == str(record.request_id)


async def test_without_spool_a_failed_write_is_only_counted(tmp_path: Path):
    durable = DurableInferenceLog(InMemoryRepository(fail_log=True), spool=None)

    assert await durable.write(_record()) is False
    assert await durable.replay() == 0


# --------------------------------------------------------------------------- #
# End to end through POST /predict
# --------------------------------------------------------------------------- #
async def test_predict_spools_row_when_database_is_down_and_replays_later(
    running_client, tmp_path: Path
):
    repository = InMemoryRepository(fail_log=True)
    spool_path = tmp_path / "spool.jsonl"
    app, _, _, _ = make_app(repository=repository, inference_log_spool_path=str(spool_path))
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    features = valid_features()
    before = await scrape(client)
    spooled_before = sample_value(before, "inference_log_spooled_total", {}) or 0
    failures_before = sample_value(before, "inference_log_failures_total", {}) or 0

    response = await client.post("/predict", json={"request_id": request_id, "features": features})

    assert response.status_code == 200, response.text  # prediction still served
    assert repository.rows == []
    after = await scrape(client)
    assert sample_value(after, "inference_log_spooled_total", {}) == spooled_before + 1
    assert sample_value(after, "inference_log_failures_total", {}) == failures_before
    assert sample_value(after, "inference_log_spool_pending", {}) == 1

    repository.fail_log = False
    assert await app.state.inference_log.replay() == 1

    assert len(repository.rows) == 1
    row = repository.rows[0]
    assert str(row.request_id) == request_id
    assert row.status == "succeeded"
    assert row.input_data == features
    assert row.prediction == "NO"
    assert (row.model_name, row.model_version) == ("diabetes", "3")
    assert row.requested_at is not None and row.requested_at.tzinfo is not None
    final = await scrape(client)
    assert sample_value(final, "inference_log_spool_pending", {}) == 0
    assert sample_value(final, "inference_log_replayed_total", {}) >= 1


async def test_startup_replays_rows_spooled_by_a_previous_process(running_client, tmp_path: Path):
    spool_path = tmp_path / "spool.jsonl"
    record = _record()
    InferenceLogSpool(spool_path).append(record)
    repository = InMemoryRepository()
    app, _, _, _ = make_app(
        repository=repository,
        inference_log_spool_path=str(spool_path),
        inference_log_replay_seconds=0.05,
    )
    await running_client(app)

    for _ in range(100):  # the replay runs file I/O in a worker thread: give it real time
        if repository.rows:
            break
        await asyncio.sleep(0.02)

    assert repository.rows == [record]
    assert not spool_path.exists()
