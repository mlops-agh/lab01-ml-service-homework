import pytest
from fastapi.testclient import TestClient

from ml_service.app import app
from ml_service.model.predictor import LABELS

client = TestClient(app)


def test_predict_returns_valid_json():
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
def test_invalid_input_returns_json_error_with_explanation(payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    detail = response.json()["detail"]
    assert len(detail) >= 1
    assert detail[0]["loc"][-1] == "text"
    assert detail[0]["msg"]


def test_malformed_json_returns_json_error():
    response = client.post(
        "/predict",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "detail" in response.json()
