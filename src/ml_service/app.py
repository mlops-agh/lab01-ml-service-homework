from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from ml_service.api.prediction import PredictRequest, PredictResponse
from ml_service.model.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources once during application startup, before the server starts accepting requests.
    app.state.predictor = Predictor.load()
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health(request: Request, response: Response) -> dict[str, str]:
        # Readiness check: 200 once the models are loaded, 503 otherwise.
        if getattr(request.app.state, "predictor", None) is None:
            response.status_code = 503
            return {"status": "loading"}
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: Request, body: PredictRequest) -> PredictResponse:
        predictor = request.app.state.predictor
        prediction = predictor.predict(body.text)
        return PredictResponse(prediction=prediction)

    return app


app = create_app()
