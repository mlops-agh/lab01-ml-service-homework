from pathlib import Path

import joblib  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).resolve().parent

LABELS: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class Predictor:
    """Wraps the embedder and classifier."""

    def __init__(self, embedder: SentenceTransformer, classifier) -> None:
        self.embedder = embedder
        self.classifier = classifier

    @classmethod
    def load(cls) -> "Predictor":
        embedder = SentenceTransformer(str(MODELS_DIR / "sentence_transformer.model"))
        classifier = joblib.load(MODELS_DIR / "classifier.joblib")
        return cls(embedder, classifier)

    def predict(self, text: str) -> str:
        # encode() takes a batch, so wrap the single text in a list
        embedding = self.embedder.encode([text])
        class_id = int(self.classifier.predict(embedding)[0])
        return LABELS[class_id]
