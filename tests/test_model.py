import pytest

from ml_service.model.predictor import LABELS, predict

SAMPLES = [
    "I love this product, it is fantastic!",
    "This is terrible, I want my money back.",
    "It arrived on Tuesday.",
]


def test_model_loads_without_errors():
    from ml_service.model.initializer import classifier, embedder

    assert hasattr(classifier, "predict")
    assert hasattr(embedder, "encode")


@pytest.mark.parametrize("text", SAMPLES)
def test_inference_returns_known_label(text):
    assert predict(text) in set(LABELS.values())
