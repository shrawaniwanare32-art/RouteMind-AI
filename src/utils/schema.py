"""Single source of truth for column names and allowed values."""

RAW_FEATURES = [
    "distance_km",
    "traffic_level",
    "temperature",
    "rain",
    "vehicle_type",
    "delivery_zone",
    "hour",
    "driver_experience",
    "current_speed",
    "route_congestion",
]

# Inputs a caller MUST provide. temperature / route_congestion are optional
# (they are imputed with the training median if missing).
REQUIRED_INPUTS = [
    "distance_km",
    "traffic_level",
    "rain",
    "vehicle_type",
    "delivery_zone",
    "hour",
    "driver_experience",
    "current_speed",
]

NUMERIC_INPUTS = [
    "distance_km",
    "traffic_level",
    "temperature",
    "rain",
    "hour",
    "driver_experience",
    "current_speed",
    "route_congestion",
]
CATEGORICAL_FEATURES = ["vehicle_type", "delivery_zone"]
ENGINEERED_FEATURES = [
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "eta_travel_min",
    "traffic_x_distance",
    "rain_x_traffic",
]
MODEL_NUMERIC = NUMERIC_INPUTS + ENGINEERED_FEATURES

TARGET_COL = "delay_minutes"
CLASS_LABELS = ["ON_TIME", "SLIGHT_DELAY", "HIGH_DELAY"]
VEHICLE_TYPES = ["bike", "scooter", "car", "van"]
DELIVERY_ZONES = ["urban", "suburban", "rural"]
PEAK_HOURS = [7, 8, 9, 17, 18, 19, 20]
