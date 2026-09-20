"""Synthetic delivery data generator.

No public dataset has GPS + traffic + weather + delay together, so we simulate one
with realistic cause-and-effect rules. Swap this for your real data later: just
put a CSV with the same columns at data/raw/deliveries.csv.
"""
import numpy as np
import pandas as pd

from src.utils.schema import PEAK_HOURS

_VEHICLE_SPEED = {"bike": 22.0, "scooter": 30.0, "car": 35.0, "van": 32.0}
_ZONE_SPEED = {"urban": 0.85, "suburban": 1.0, "rural": 1.1}
_ZONE_TRAFFIC = {"urban": 4.5, "suburban": 3.2, "rural": 1.8}


def generate_deliveries(
    n: int, seed: int = 42, drift: bool = False, missing_rate: float = 0.0
) -> pd.DataFrame:
    """Create `n` deliveries. `drift=True` shifts the distributions (more traffic,
    more rain, longer trips) so you can demo data-drift detection."""
    rng = np.random.default_rng(seed)

    zone = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.55, 0.30, 0.15])
    vehicle = rng.choice(["bike", "scooter", "car", "van"], size=n, p=[0.35, 0.30, 0.25, 0.10])

    hour_w = np.ones(24)
    hour_w[0:6] = 0.15
    hour_w[PEAK_HOURS] = 3.0
    hour = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    peak = np.isin(hour, PEAK_HOURS).astype(float)

    distance = np.clip(rng.gamma(3.2 if drift else 2.5, 3.8 if drift else 3.0, n), 0.5, 60)
    traffic = (
        np.array([_ZONE_TRAFFIC[z] for z in zone])
        + 2.2 * peak
        + rng.normal(1.5 if drift else 0.0, 1.3, n)
    )
    traffic = np.round(np.clip(traffic, 0, 10), 1)

    temperature = np.round(np.clip(rng.normal(28, 6, n), 5, 45), 1)
    rain = (rng.random(n) < (0.35 if drift else 0.20)).astype(int)
    experience = np.round(np.clip(rng.gamma(2.0, 1.8, n), 0.1, 15), 1)

    speed = (
        np.array([_VEHICLE_SPEED[v] for v in vehicle])
        * np.array([_ZONE_SPEED[z] for z in zone])
        * (1 - 0.06 * traffic)
        * np.where(rain == 1, 0.85, 1.0)
        + rng.normal(0, 3, n)
    )
    speed = np.round(np.clip(speed, 3, 80), 1)
    congestion = np.round(np.clip(0.6 * traffic + rng.normal(0, 1.5, n), 0, 10), 1)

    delay = (
        -9.0
        + 0.9 * distance * (traffic / 10.0)
        + 0.35 * distance
        + 1.9 * np.maximum(traffic - 4, 0)
        + 6.5 * rain * np.where(vehicle == "bike", 1.6, 1.0)
        + 5.5 * peak
        + 1.7 * congestion
        + 0.9 * np.maximum(22 - speed, 0)
        - 1.4 * np.minimum(experience, 6)
        + np.where(zone == "rural", 3.0, 0.0)
        + rng.normal(0, 3.5, n)
    )
    delay = np.round(np.clip(delay, 0, None), 1)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{10000 + i}" for i in range(n)],
            "distance_km": np.round(distance, 2),
            "traffic_level": traffic,
            "temperature": temperature,
            "rain": rain,
            "vehicle_type": vehicle,
            "delivery_zone": zone,
            "hour": hour,
            "driver_experience": experience,
            "current_speed": speed,
            "route_congestion": congestion,
            "delay_minutes": delay,
        }
    )

    if missing_rate > 0:
        for col in ["temperature", "driver_experience", "current_speed", "route_congestion"]:
            df.loc[rng.random(n) < missing_rate, col] = np.nan
    return df
