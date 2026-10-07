import pytest

from ml_service.model.predictor import LABELS, Predictor

SAMPLES = [
    "I love this product, it is fantastic!",
    "This is terrible, I want my money back.",
    "It arrived on Tuesday.",
]


def test_model_loads_without_errors(predictor: Predictor) -> None:
    assert hasattr(predictor.classifier, "predict")
    assert hasattr(predictor.embedder, "encode")


@pytest.mark.parametrize("text", SAMPLES)
def test_inference_returns_known_label(predictor: Predictor, text: str) -> None:
    assert predictor.predict(text) in set(LABELS.values())
