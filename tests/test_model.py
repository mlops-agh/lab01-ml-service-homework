import pytest

from ml_service.model.predictor import LABELS

SAMPLES = [
    "I love this product, it is fantastic!",
    "This is terrible, I want my money back.",
    "It arrived on Tuesday.",
]


def test_model_loads_without_errors(predictor):
    assert hasattr(predictor.classifier, "predict")
    assert hasattr(predictor.embedder, "encode")


@pytest.mark.parametrize("text", SAMPLES)
def test_inference_returns_known_label(predictor, text):
    assert predictor.predict(text) in set(LABELS.values())
