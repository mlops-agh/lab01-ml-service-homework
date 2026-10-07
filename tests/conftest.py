import pytest
from fastapi.testclient import TestClient

from ml_service.app import app
from ml_service.model.predictor import Predictor


@pytest.fixture(scope="session")
def predictor():
    return Predictor.load()


@pytest.fixture(scope="session")
def client():
    # `with` runs the lifespan, so the models are loaded once per session
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_models():
    # without `with` the lifespan does not run: no models, fast tests
    return TestClient(app)
