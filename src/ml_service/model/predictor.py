from ml_service.model.initializer import classifier, embedder

LABELS: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def predict(text: str) -> str:
    embedding = embedder.encode([text])
    class_id = int(classifier.predict(embedding)[0])
    return LABELS[class_id]
