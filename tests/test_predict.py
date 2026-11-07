"""T2: POST /predict contract, inference path, and inference.logs rows."""

from __future__ import annotations

import uuid

import pytest
from diabetes_model.contract import INPUT_COLUMNS

from conftest import FakeModel, FakeRegistry, InMemoryRepository, make_app, valid_features


def test_predict_endpoint_is_exposed_and_uses_fixture_model(test_client, prediction_model):
    openapi = test_client.get("/openapi.json").json()
    assert "post" in openapi["paths"]["/predict"]

    response = test_client.post("/predict", json={"features": valid_features()})

    assert response.status_code == 200, response.text
    assert response.json()["prediction"] == "NO"
    assert response.json()["probabilities"] == {"<30": 0.05, ">30": 0.15, "NO": 0.8}
    assert len(prediction_model.frames) == 1


async def test_predict_success_returns_contract_and_logs_row(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    request_id = str(uuid.uuid4())
    features = valid_features()

    response = await client.post("/predict", json={"request_id": request_id, "features": features})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == request_id
    assert body["prediction"] == "NO"
    assert body["probabilities"] == {"<30": 0.05, ">30": 0.15, "NO": 0.8}
    assert body["model"] == {"name": "diabetes", "version": "3", "alias": "production"}
    assert body["processing_time_ms"] >= 0

    assert len(repository.rows) == 1
    row = repository.rows[0]
    assert str(row.request_id) == request_id
    assert row.status == "succeeded"
    assert row.input_data == features  # exact submitted object
    assert row.prediction == "NO"
    assert row.probabilities == {"<30": 0.05, ">30": 0.15, "NO": 0.8}
    assert (row.model_name, row.model_version, row.model_alias) == ("diabetes", "3", "production")
    assert row.processing_time_ms == body["processing_time_ms"]
    assert row.error_code is None
    assert row.requested_at is not None and row.requested_at.tzinfo is not None


async def test_predict_generates_request_id_when_omitted(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)

    body = (await client.post("/predict", json={"features": valid_features()})).json()

    generated = uuid.UUID(body["request_id"])
    assert repository.rows[0].request_id == generated


async def test_predict_feeds_normalised_frame_to_model(running_client):
    model = FakeModel()
    app, _, _, _ = make_app(registry=FakeRegistry(model=model))
    client = await running_client(app)

    await client.post("/predict", json={"features": valid_features()})

    frame = model.frames[0]
    assert list(frame.columns) == list(INPUT_COLUMNS)
    assert len(frame) == 1
    assert frame.loc[0, "time_in_hospital"] == 6.0
    assert frame.loc[0, "payer_code"] is None


@pytest.mark.parametrize("extra_key", ["encounter_id", "patient_nbr", "readmitted", "weight", "examide", "citoglipton", "unknown_field"])
async def test_predict_rejects_identifiers_and_unknown_keys(running_client, extra_key):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    features = valid_features()
    features[extra_key] = "x"

    response = await client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert extra_key in response.text
    assert len(repository.rows) == 1
    row = repository.rows[0]
    assert row.status == "failed"
    assert row.error_code == "validation_error"
    assert row.input_data == features
    assert extra_key in (row.error_message or "")
    assert row.prediction is None and row.model_version is None


async def test_predict_rejects_missing_feature(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)
    features = valid_features()
    del features["insulin"]

    response = await client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert repository.rows[0].error_code == "validation_error"


async def test_predict_rejects_wrong_types(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)

    numeric_as_string = valid_features()
    numeric_as_string["time_in_hospital"] = "6"
    assert (await client.post("/predict", json={"features": numeric_as_string})).status_code == 422

    code_as_number = valid_features()
    code_as_number["admission_type_id"] = 1
    assert (await client.post("/predict", json={"features": code_as_number})).status_code == 422


async def test_predict_accepts_nulls_and_integer_numbers(running_client):
    app, _, _, _ = make_app()
    client = await running_client(app)
    features = valid_features()
    features["race"] = None
    features["number_outpatient"] = None
    features["num_medications"] = 3.5

    assert (await client.post("/predict", json={"features": features})).status_code == 200


async def test_predict_malformed_json_logs_raw_body(running_client):
    app, _, repository, _ = make_app()
    client = await running_client(app)

    response = await client.post(
        "/predict", content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert repository.rows[0].status == "failed"
    assert "raw" in repository.rows[0].input_data


async def test_predict_503_when_model_unavailable_and_logs_failure(running_client):
    app, _, repository, _ = make_app(registry=FakeRegistry(version=None))
    client = await running_client(app)
    features = valid_features()

    response = await client.post("/predict", json={"features": features})

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "model_unavailable"
    row = repository.rows[0]
    assert row.status == "failed"
    assert row.error_code == "model_unavailable"
    assert row.input_data == features


async def test_predict_500_on_inference_error(running_client):
    class BrokenModel(FakeModel):
        def predict(self, frame):
            raise ValueError("boom")

    app, _, repository, _ = make_app(registry=FakeRegistry(model=BrokenModel()))
    client = await running_client(app)

    response = await client.post("/predict", json={"features": valid_features()})

    assert response.status_code == 500
    assert response.json()["detail"]["error_code"] == "inference_error"
    assert repository.rows[0].error_code == "inference_error"
    assert "boom" in repository.rows[0].error_message


async def test_predict_still_answers_when_log_write_fails(running_client):
    app, _, _, _ = make_app(repository=InMemoryRepository(fail_log=True))
    client = await running_client(app)

    response = await client.post("/predict", json={"features": valid_features()})

    assert response.status_code == 200
    assert response.json()["prediction"] == "NO"


def test_feature_schema_exposes_all_44_contract_keys():
    from api_inference.schemas import PredictFeatures

    properties = PredictFeatures.model_json_schema()["properties"]
    assert set(properties) == set(INPUT_COLUMNS)
    assert "glyburide-metformin" in properties
    assert PredictFeatures.model_config.get("extra") == "forbid"
