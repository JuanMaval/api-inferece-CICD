"""POST /predict/{request_id}/feedback: ground truth for a logged prediction."""

from __future__ import annotations

import uuid

import pytest

from conftest import InMemoryRepository, make_app, valid_features


async def _predict(client, request_id: str) -> None:
    response = await client.post(
        "/predict", json={"request_id": request_id, "features": valid_features()}
    )
    assert response.status_code == 200, response.text


async def test_feedback_records_label_for_logged_prediction(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    await _predict(client, request_id)

    response = await client.post(
        f"/predict/{request_id}/feedback", json={"readmitted": "<30", "source": "ehr"}
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "request_id": request_id,
        "readmitted": "<30",
        "prediction": "NO",
        "created": True,
    }
    stored = repository.feedback[uuid.UUID(request_id)]
    assert (stored.readmitted, stored.source) == ("<30", "ehr")


async def test_feedback_second_call_replaces_label(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    await _predict(client, request_id)
    await client.post(f"/predict/{request_id}/feedback", json={"readmitted": "<30"})

    response = await client.post(f"/predict/{request_id}/feedback", json={"readmitted": ">30"})

    assert response.status_code == 200, response.text
    assert response.json()["created"] is False
    assert response.json()["readmitted"] == ">30"
    assert repository.feedback[uuid.UUID(request_id)].readmitted == ">30"
    assert repository.feedback[uuid.UUID(request_id)].source is None


async def test_feedback_for_failed_request_has_null_prediction(running_client):
    """A label can still be attached to a request the model failed to score."""
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    response = await client.post(
        "/predict", json={"request_id": request_id, "features": {"race": "Caucasian"}}
    )
    assert response.status_code == 422
    assert repository.rows[0].status == "failed"

    response = await client.post(f"/predict/{request_id}/feedback", json={"readmitted": "NO"})

    assert response.status_code == 200, response.text
    assert response.json()["prediction"] is None


async def test_feedback_unknown_request_is_404(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())

    response = await client.post(f"/predict/{request_id}/feedback", json={"readmitted": "NO"})

    assert response.status_code == 404, response.text
    body = response.json()
    assert body["request_id"] == request_id
    assert body["detail"]["error_code"] == "unknown_request"
    assert repository.feedback == {}
    assert repository.rows == []  # not an inference request: no inference.logs row


@pytest.mark.parametrize(
    "body",
    [
        {"readmitted": "yes"},
        {"readmitted": None},
        {},
        {"readmitted": "NO", "extra": 1},
        {"readmitted": "NO", "source": "x" * 201},
    ],
)
async def test_feedback_rejects_invalid_body(running_client, body):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    await _predict(client, request_id)

    response = await client.post(f"/predict/{request_id}/feedback", json=body)

    assert response.status_code == 422, response.text
    assert repository.feedback == {}
    assert len(repository.rows) == 1  # the 422 handler logs /predict bodies only


async def test_feedback_not_a_uuid_is_422(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    response = await client.post("/predict/not-a-uuid/feedback", json={"readmitted": "NO"})

    assert response.status_code == 422


async def test_feedback_returns_503_when_database_write_fails(running_client):
    app, _, repository, _ = make_app(repository=InMemoryRepository(fail_feedback=True))
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    await _predict(client, request_id)

    response = await client.post(f"/predict/{request_id}/feedback", json={"readmitted": "NO"})

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "database_unavailable"
    assert repository.feedback == {}
