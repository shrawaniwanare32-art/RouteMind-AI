from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary", summary="Model health: risk mix, latency, drift, live accuracy")
def summary(request: Request):
    return request.app.state.monitoring.summary()


@router.get("/drift", summary="Per-feature PSI drift report")
def drift(request: Request):
    return request.app.state.monitoring.drift_report()


@router.get("/recent", summary="Recent predictions (optionally only HIGH/CRITICAL)")
def recent(request: Request, limit: int = Query(50, ge=1, le=500), risky_only: bool = False):
    return request.app.state.monitoring.recent_predictions(limit=limit, risky_only=risky_only)
