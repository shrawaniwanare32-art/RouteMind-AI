import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained):
    from api.main import app
    with TestClient(app) as c:   # runs the lifespan (loads the model)
        yield c


PAYLOAD = {
    "order_id": "ORD-10231", "distance_km": 18.4, "traffic_level": 8, "temperature": 31,
    "rain": True, "vehicle_type": "bike", "delivery_zone": "urban", "hour": 18,
    "driver_experience": 2.4, "current_speed": 18,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["order_id"] == "ORD-10231"
    assert body["risk_level"] in {"HIGH", "CRITICAL"}
    assert body["recommended_action"] in {"REROUTE", "PRIORITIZE_AND_NOTIFY"}


def test_predict_accepts_original_distance_alias(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "distance_km"} | {"distance": 18.4}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("change", [
    {"traffic_level": 42},
    {"vehicle_type": "helicopter"},
    {"hour": 30},
    {"distance_km": -1},
])
def test_predict_rejects_invalid_input(client, change):
    assert client.post("/predict", json=PAYLOAD | change).status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "traffic_level"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"records": [PAYLOAD, PAYLOAD | {"order_id": "B2"}]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_feedback_and_monitoring_endpoints(client):
    client.post("/predict", json=PAYLOAD)
    fb = client.post("/feedback", json={"order_id": "ORD-10231", "actual_delay_minutes": 35})
    assert fb.status_code == 200
    s = client.get("/monitoring/summary")
    assert s.status_code == 200 and s.json()["n_predictions"] >= 1
    assert client.get("/monitoring/drift").status_code == 200
    assert client.get("/monitoring/recent", params={"limit": 5}).status_code == 200


def test_reload_model(client):
    r = client.post("/model/reload")
    assert r.status_code == 200 and r.json()["status"] == "reloaded"
