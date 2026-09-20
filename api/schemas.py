"""Request / response models (validated by Pydantic)."""
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DeliveryRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "order_id": "ORD-10231",
                "distance_km": 18.4,
                "traffic_level": 8,
                "temperature": 31,
                "rain": True,
                "vehicle_type": "bike",
                "delivery_zone": "urban",
                "hour": 18,
                "driver_experience": 2.4,
                "current_speed": 18,
            }
        },
    )

    order_id: str | None = Field(None, max_length=64)
    # "distance" is accepted as an alias so the original example payload works too
    distance_km: float = Field(..., gt=0, le=200, validation_alias=AliasChoices("distance_km", "distance"))
    traffic_level: float = Field(..., ge=0, le=10)
    temperature: float | None = Field(None, ge=-10, le=55)
    rain: bool
    vehicle_type: Literal["bike", "scooter", "car", "van"]
    delivery_zone: Literal["urban", "suburban", "rural"]
    hour: int = Field(..., ge=0, le=23)
    driver_experience: float = Field(..., ge=0, le=45)
    current_speed: float = Field(..., ge=0, le=120)
    route_congestion: float | None = Field(None, ge=0, le=10)


class RiskFactor(BaseModel):
    factor: str
    impact_minutes: float
    contribution_pct: float


class PredictionResponse(BaseModel):
    order_id: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    delay_probability: float
    predicted_class: str
    class_probabilities: dict[str, float]
    expected_delay_minutes: float
    risk_factors: list[RiskFactor]
    recommended_action: str
    recommended_actions: list[str]
    model_version: str


class BatchRequest(BaseModel):
    records: list[DeliveryRequest] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    order_id: str
    actual_delay_minutes: float = Field(..., ge=0, le=600)
