"""RouteMind AI - FastAPI service.

Run:  uvicorn api.main:app --reload --port 8000     (docs at /docs)
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import monitoring, prediction
from src.components.monitoring import MonitoringService, PredictionLogger
from src.config.configuration import load_config
from src.models.model_loader import ModelRegistry
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.exception import CustomException, PredictionException
from src.utils.logger import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    if ModelRegistry(cfg).latest_version() is None:
        if os.getenv("ROUTEMIND_AUTO_TRAIN", "true").lower() == "true":
            logger.warning("No model found - training one now (first start)")
            TrainingPipeline(cfg).run()
        else:
            raise RuntimeError("No trained model found and ROUTEMIND_AUTO_TRAIN=false")
    app.state.prediction_logger = PredictionLogger(cfg)
    app.state.predictor = PredictionPipeline(cfg, prediction_logger=app.state.prediction_logger)
    app.state.monitoring = MonitoringService(cfg)
    logger.info("API ready - model %s", app.state.predictor.version)
    yield


app = FastAPI(
    title="RouteMind AI",
    description="Predictive Delivery Disruption & Recovery System",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(prediction.router)
app.include_router(monitoring.router)


@app.get("/", tags=["system"])
def root():
    return {"service": "RouteMind AI", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.predictor.version}


@app.exception_handler(PredictionException)
async def prediction_error_handler(request: Request, exc: PredictionException):
    logger.error("%s", exc)
    return JSONResponse(status_code=422, content=exc.to_dict())


@app.exception_handler(CustomException)
async def custom_error_handler(request: Request, exc: CustomException):
    logger.error("%s", exc)
    return JSONResponse(status_code=500, content=exc.to_dict())
