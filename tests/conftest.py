import pytest
from fastapi.testclient import TestClient

from ml_service.app import create_app
from ml_service.model.predictor import Predictor


@pytest.fixture(scope="session")
def predictor():
    return Predictor.load()


@pytest.fixture(scope="session")
def client():
    # `with` runs the lifespan, so the models are loaded once per session
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def client_no_models():
    # fresh app without `with`: lifespan does not run, app.state is empty, no models loaded
    return TestClient(create_app())
