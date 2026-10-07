import pytest
from starlette.testclient import TestClient

from ml_service.app import app
from ml_service.model.predictor import LABELS


def test_health_ok_after_startup(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_loading_before_startup(
    client_no_models: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delattr(app.state, "predictor", raising=False)
    response = client_no_models.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_predict_returns_valid_json(client: TestClient):
    response = client.post("/predict", json={"text": "I love this!"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"prediction"}
    assert body["prediction"] in set(LABELS.values())


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing field
        {"text": ""},  # empty string
        {"text": 123},  # wrong type
        {"text": None},  # null
    ],
)
def test_invalid_input_returns_json_error_with_explanation(
    client_no_models: TestClient, payload
):
    response = client_no_models.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    detail = response.json()["detail"]
    assert len(detail) >= 1
    assert detail[0]["loc"][-1] == "text"
    assert detail[0]["msg"]


def test_malformed_json_returns_json_error(client_no_models: TestClient):
    response = client_no_models.post(
        "/predict",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "detail" in response.json()
