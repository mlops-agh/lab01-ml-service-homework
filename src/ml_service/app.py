from fastapi import FastAPI
from ml_service.api.prediction import PredictRequest, PredictResponse

app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="positive")
