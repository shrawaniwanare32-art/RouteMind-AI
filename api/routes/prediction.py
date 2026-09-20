from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    BatchRequest,
    BatchResponse,
    DeliveryRequest,
    FeedbackRequest,
    PredictionResponse,
)

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse, summary="Predict delay risk for one delivery")
def predict(req: DeliveryRequest, request: Request):
    result = request.app.state.predictor.predict([req.model_dump()])[0]
    return result


@router.post("/predict/batch", response_model=BatchResponse, summary="Predict for up to 500 deliveries")
def predict_batch(req: BatchRequest, request: Request):
    results = request.app.state.predictor.predict([r.model_dump() for r in req.records])
    return {"count": len(results), "predictions": results}


@router.post("/feedback", summary="Report the real delay of a finished delivery")
def feedback(req: FeedbackRequest, request: Request):
    """Ground truth feeds the live-accuracy monitor and the retraining trigger."""
    request.app.state.prediction_logger.log_feedback(req.order_id, req.actual_delay_minutes)
    return {"status": "recorded", "order_id": req.order_id}


@router.post("/model/reload", summary="Load the newest production model without restarting")
def reload_model(request: Request):
    try:
        version = request.app.state.predictor.reload()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "reloaded", "model_version": version}
