"""T3: Prometheus middleware and GET /metrics."""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from conftest import FakeRegistry, InMemoryRepository, make_app, valid_features


def sample_value(text: str, name: str, labels: dict[str, str]) -> float | None:
    """Return the value of one sample or None when the series is absent."""
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


async def scrape(client) -> str:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.text


async def test_metrics_count_requests_by_route_template_and_status(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)
    before = await scrape(client)
    health_before = sample_value(before, "http_requests_total", {"method": "GET", "path": "/health", "status": "200"}) or 0
    predict_before = sample_value(before, "http_requests_total", {"method": "POST", "path": "/predict", "status": "200"}) or 0

    await client.get("/health")
    await client.post("/predict", json={"features": valid_features()})
    after = await scrape(client)

    assert sample_value(after, "http_requests_total", {"method": "GET", "path": "/health", "status": "200"}) == health_before + 1
    assert sample_value(after, "http_requests_total", {"method": "POST", "path": "/predict", "status": "200"}) == predict_before + 1
    assert sample_value(after, "http_request_duration_seconds_count", {"method": "POST", "path": "/predict"}) >= 1
    assert sample_value(after, "http_requests_in_progress", {}) == 0
    # /metrics itself is never counted.
    assert sample_value(after, "http_requests_total", {"path": "/metrics"}) is None


async def test_metrics_label_unknown_paths_with_bounded_value(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    await client.get("/does-not-exist/42")
    text = await scrape(client)

    assert sample_value(text, "http_requests_total", {"path": "/does-not-exist/42"}) is None
    assert sample_value(text, "http_requests_total", {"path": "unmatched", "status": "404"}) >= 1


async def test_metrics_track_prediction_outcomes(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)
    before = await scrape(client)
    ok_before = sample_value(before, "predictions_total", {"status": "succeeded"}) or 0
    invalid_before = sample_value(before, "predictions_total", {"status": "validation_error"}) or 0

    await client.post("/predict", json={"features": valid_features()})
    await client.post("/predict", json={"features": {"encounter_id": 1}})
    after = await scrape(client)

    assert sample_value(after, "predictions_total", {"status": "succeeded"}) == ok_before + 1
    assert sample_value(after, "predictions_total", {"status": "validation_error"}) == invalid_before + 1
    assert sample_value(after, "prediction_duration_seconds_count", {}) >= 1


async def test_metrics_track_model_unavailable_and_log_failures(running_client):
    app, _, _, _ = make_app(registry=FakeRegistry(version=None), repository=InMemoryRepository(fail_log=True))
    client = await running_client(app)
    before = await scrape(client)
    unavailable_before = sample_value(before, "predictions_total", {"status": "model_unavailable"}) or 0
    failures_before = sample_value(before, "inference_log_failures_total", {}) or 0

    await client.post("/predict", json={"features": valid_features()})
    after = await scrape(client)

    assert sample_value(after, "predictions_total", {"status": "model_unavailable"}) == unavailable_before + 1
    assert sample_value(after, "inference_log_failures_total", {}) == failures_before + 1


async def test_model_info_gauge_follows_reloads(running_client):
    app, registry, _, _ = make_app()
    client = await running_client(app)
    text = await scrape(client)
    assert sample_value(text, "model_info", {"name": "diabetes", "version": "3", "alias": "production"}) == 1

    registry.version = "4"
    await client.post("/model/reload")
    text = await scrape(client)

    assert sample_value(text, "model_info", {"version": "4"}) == 1
    assert sample_value(text, "model_info", {"version": "3"}) is None
    assert sample_value(text, "model_reloads_total", {"result": "success"}) >= 2


async def test_model_reload_failure_is_counted(running_client):
    app, registry, _, _ = make_app()
    client = await running_client(app)
    before = sample_value(await scrape(client), "model_reloads_total", {"result": "failure"}) or 0
    registry.fail_resolve = True

    await client.post("/model/reload")

    assert sample_value(await scrape(client), "model_reloads_total", {"result": "failure"}) == before + 1


async def test_metrics_include_python_platform_collector(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    text = await scrape(client)

    assert "python_info" in text
