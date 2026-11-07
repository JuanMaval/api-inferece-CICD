"""T1: /health, /model-info, POST /model/reload and the ModelService lifecycle."""

from __future__ import annotations

import pytest

from conftest import FakeModel, FakeRegistry, InMemoryRepository, make_app, make_service, make_settings


async def test_health_reports_model_and_database(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "database": "ok"}


async def test_health_stays_200_when_model_unavailable(running_client):
    app, registry, _, _ = make_app(registry=FakeRegistry(version=None))
    client = await running_client(app)

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    # Startup retried the bounded number of times and then gave up.
    assert registry.resolve_calls == 2


async def test_health_reports_database_error(running_client):
    app, _, _, _ = make_app(repository=InMemoryRepository(fail_ping=True))
    client = await running_client(app)

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "error"


async def test_model_info_when_ready(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    body = (await client.get("/model-info")).json()

    assert body["status"] == "ready"
    assert body["name"] == "diabetes"
    assert body["alias"] == "production"
    assert body["version"] == "3"
    assert body["run_id"] == "run-3"
    assert body["loaded_at"] is not None
    assert body["refresh_seconds"] == 0
    assert body["classes"] == ["<30", ">30", "NO"]


async def test_model_info_when_unavailable(running_client):
    app, _, _, _ = make_app(registry=FakeRegistry(version=None))
    client = await running_client(app)

    body = (await client.get("/model-info")).json()

    assert body["status"] == "unavailable"
    assert body["version"] is None
    assert body["loaded_at"] is None
    assert "no version" in body["last_error"]


async def test_reload_swaps_to_new_alias_version(running_client):
    app, registry, _, _ = make_app()
    client = await running_client(app)
    registry.version = "4"
    registry.model = FakeModel(label="<30")

    response = await client.post("/model/reload")

    assert response.status_code == 200
    assert response.json() == {
        "previous_version": "3",
        "current_version": "4",
        "changed": True,
    }
    assert (await client.get("/model-info")).json()["version"] == "4"


async def test_reload_same_version_reports_unchanged(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    body = (await client.post("/model/reload")).json()

    assert body == {"previous_version": "3", "current_version": "3", "changed": False}


async def test_reload_failure_returns_503_and_keeps_previous(running_client):
    app, registry, _, _ = make_app()
    client = await running_client(app)
    registry.fail_resolve = True

    response = await client.post("/model/reload")

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "model_load_error"
    assert (await client.get("/model-info")).json()["version"] == "3"


async def test_refresh_if_changed_only_swaps_on_new_version():
    registry = FakeRegistry(version="3")
    service = make_service(registry, make_settings())
    await service.load()

    assert await service.refresh_if_changed() is False
    assert registry.load_calls == 1

    registry.version = "5"
    assert await service.refresh_if_changed() is True
    assert service.loaded is not None
    assert service.loaded.version == "5"
    assert registry.load_calls == 2
    assert service.info().last_check_at is not None


async def test_refresh_failure_keeps_current_model_and_records_error():
    registry = FakeRegistry(version="3")
    service = make_service(registry, make_settings())
    await service.load()
    registry.fail_resolve = True

    assert await service.refresh_if_changed() is False
    assert service.loaded is not None and service.loaded.version == "3"
    assert service.info().status == "ready"
    assert service.info().last_error is not None


async def test_predict_before_load_raises_not_loaded():
    from api_inference.model import ModelNotLoadedError

    service = make_service(FakeRegistry(), make_settings())

    with pytest.raises(ModelNotLoadedError):
        await service.predict({"age": "[70-80)"})
