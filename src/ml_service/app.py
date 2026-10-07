from fastapi import FastAPI
from ml_service.api.prediction import PredictRequest, PredictResponse
from ml_service.model import predictor

app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = predictor.predict(request.text)
    return PredictResponse(prediction=prediction)
