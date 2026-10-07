import joblib  # type: ignore[import-untyped]
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).resolve().parent

embedder = SentenceTransformer(str(MODELS_DIR / "sentence_transformer.model"))

classifier = joblib.load(MODELS_DIR / "classifier.joblib")
