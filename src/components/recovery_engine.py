"""Model 3 - rule-based recovery recommendation.

Turns (risk level + main risk factor) into an operational action. Starts as a rule
engine; later you can replace `recommend()` with an ML/bandit model.
"""

RISK_ACTIONS = {
    "LOW": ("CONTINUE", ["Continue current route"]),
    "MEDIUM": ("MONITOR", ["Monitor route progress", "Re-check risk in 10 minutes"]),
    "HIGH": (
        "REROUTE",
        ["Reroute driver via alternate route", "Notify customer of possible delay"],
    ),
    "CRITICAL": (
        "PRIORITIZE_AND_NOTIFY",
        [
            "Prioritize this shipment",
            "Notify customer immediately with a new ETA",
            "Consider reassigning to the nearest available driver",
        ],
    ),
}

FACTOR_TIPS = {
    "Rain": "Bike/scooter in rain: consider switching to a car or van",
    "Heavy traffic": "Avoid main arterial roads on the reroute",
    "Route congestion": "Pick a route that avoids the congested segment",
    "Peak hour": "Batch nearby orders to save trips during peak hour",
    "Low current speed": "Check for a vehicle problem or road block",
    "Inexperienced driver": "Send turn-by-turn guidance to the driver",
    "Long distance": "Consider splitting the delivery to a closer hub",
}


def risk_level(delay_probability: float, expected_delay_min: float, risk_cfg: dict) -> str:
    """LOW / MEDIUM / HIGH / CRITICAL from P(delay) and expected delay minutes."""
    unlikely = delay_probability < risk_cfg["medium_probability"]
    negligible = expected_delay_min < risk_cfg["low_delay_minutes"]
    if unlikely or negligible:
        return "LOW"
    if expected_delay_min >= risk_cfg["critical_delay_minutes"]:
        return "CRITICAL"
    if expected_delay_min >= risk_cfg["high_delay_minutes"]:
        return "HIGH"
    return "MEDIUM"


def recommend(level: str, risk_factors: list[dict], vehicle_type: str | None = None) -> tuple[str, list[str]]:
    code, actions = RISK_ACTIONS[level]
    actions = list(actions)
    if level in ("HIGH", "CRITICAL") and risk_factors:
        top = risk_factors[0]["factor"]
        tip = FACTOR_TIPS.get(top)
        if top == "Rain" and vehicle_type not in ("bike", "scooter"):
            tip = None  # the tip only makes sense for two-wheelers
        if tip:
            actions.append(tip)
    return code, actions
