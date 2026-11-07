"""Shared fakes: an in-memory model registry, a fake sklearn model, and an
in-memory inference log repository. No MLflow server, no Postgres."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx
import numpy as np
import pytest
from diabetes_model.contract import INPUT_COLUMNS, NUMERIC_INPUTS
from fastapi.testclient import TestClient

from api_inference.main import create_app
from api_inference.model import ModelLoadError, ModelService, RegisteredVersion
from api_inference.repository import FeedbackRecord, InferenceLogRecord, LoggedRequest
from api_inference.settings import Settings

# sklearn sorts classes lexicographically, so this is the realistic order.
FAKE_CLASSES = np.array(["<30", ">30", "NO"], dtype=object)


class FakeModel:
    """Minimal stand-in for the served sklearn pipeline."""

    classes_ = FAKE_CLASSES

    def __init__(self, label: str = "NO", proba: tuple[float, ...] = (0.05, 0.15, 0.8)):
        self.label = label
        self.proba = proba
        self.frames: list[Any] = []

    def predict(self, frame):
        self.frames.append(frame)
        return np.array([self.label] * len(frame), dtype=object)

    def predict_proba(self, frame):
        return np.array([list(self.proba)] * len(frame), dtype="float64")


class FakeRegistry:
    """Fake MLflow registry: resolves an alias to a version and loads a model."""

    def __init__(self, version: str | None = "3", model: Any | None = None):
        self.version = version
        self.model = model or FakeModel()
        self.resolve_calls = 0
        self.load_calls = 0
        self.fail_resolve = version is None

    def resolve(self, name: str, alias: str) -> RegisteredVersion:
        self.resolve_calls += 1
        if self.fail_resolve or self.version is None:
            raise ModelLoadError(f"no version for {name}@{alias}")
        return RegisteredVersion(
            name=name, alias=alias, version=self.version, run_id=f"run-{self.version}"
        )

    def load(self, model_uri: str) -> Any:
        self.load_calls += 1
        return self.model


@dataclass
class InMemoryRepository:
    rows: list[InferenceLogRecord] = field(default_factory=list)
    feedback: dict[UUID, FeedbackRecord] = field(default_factory=dict)
    fail_ping: bool = False
    fail_log: bool = False
    fail_feedback: bool = False
    opened: bool = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.opened = False

    async def ping(self) -> None:
        if self.fail_ping:
            raise RuntimeError("database down")

    async def log(self, record: InferenceLogRecord) -> None:
        if self.fail_log:
            raise RuntimeError("insert failed")
        self.rows.append(record)

    async def get_log(self, request_id: UUID) -> LoggedRequest | None:
        for row in self.rows:
            if row.request_id == request_id:
                return LoggedRequest(
                    request_id=row.request_id, status=row.status, prediction=row.prediction
                )
        return None

    async def upsert_feedback(self, record: FeedbackRecord) -> bool:
        if self.fail_feedback:
            raise RuntimeError("upsert failed")
        created = record.request_id not in self.feedback
        self.feedback[record.request_id] = record
        return created


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "mlflow_tracking_uri": "http://mlflow.test",
        "model_name": "diabetes",
        "model_alias": "production",
        "model_refresh_seconds": 0,  # background refresh disabled in tests
        "model_load_retries": 2,
        "model_load_backoff_seconds": 0.0,
        "train_database_url": "postgresql://app:app@localhost:5432/diabetes",
        # Spool disabled by default so a failing fake repository behaves as
        # before (row counted as lost); spool tests pass a tmp path explicitly.
        "inference_log_spool_path": "",
        "inference_log_replay_seconds": 0,
    }
    base.update(overrides)
    return Settings(**base)


def make_service(registry: FakeRegistry, settings: Settings) -> ModelService:
    return ModelService(
        name=settings.model_name,
        alias=settings.model_alias,
        refresh_seconds=settings.model_refresh_seconds,
        resolve_version=registry.resolve,
        load_model=registry.load,
    )


def make_app(
    registry: FakeRegistry | None = None,
    repository: InMemoryRepository | None = None,
    **settings_overrides: Any,
):
    settings = make_settings(**settings_overrides)
    registry = registry or FakeRegistry()
    repository = repository or InMemoryRepository()
    service = make_service(registry, settings)
    app = create_app(settings=settings, model_service=service, repository=repository)
    return app, registry, repository, service


@pytest.fixture
def prediction_model() -> FakeModel:
    """Provide an isolated model instance for synchronous endpoint tests."""
    return FakeModel()


@pytest.fixture
def test_client(prediction_model: FakeModel) -> Iterator[TestClient]:
    app, _, _, _ = make_app(registry=FakeRegistry(model=prediction_model))
    with TestClient(app) as client:
        yield client


def valid_features() -> dict[str, Any]:
    features: dict[str, Any] = {column: "No" for column in INPUT_COLUMNS}
    features.update(
        {
            "race": "Caucasian",
            "gender": "Male",
            "age": "[70-80)",
            "admission_type_id": "1",
            "discharge_disposition_id": "1",
            "admission_source_id": "7",
            "payer_code": None,
            "medical_specialty": None,
            "diag_1": "530",
            "diag_2": "530",
            "diag_3": "787",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "change": "No",
            "diabetesMed": "No",
        }
    )
    for column in NUMERIC_INPUTS:
        features[column] = 0
    features["time_in_hospital"] = 6
    features["num_lab_procedures"] = 13
    features["num_procedures"] = 3
    features["num_medications"] = 3
    features["number_diagnoses"] = 9
    return features


@pytest.fixture
async def running_client():
    """Factory fixture: builds an app, runs its lifespan, yields an HTTP client."""
    stack: list[Any] = []

    async def _start(app) -> httpx.AsyncClient:
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        stack.append((lifespan, client))
        return client

    yield _start

    for lifespan, client in reversed(stack):
        await client.aclose()
        await lifespan.__aexit__(None, None, None)
