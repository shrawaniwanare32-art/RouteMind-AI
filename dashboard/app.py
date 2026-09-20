"""RouteMind AI - Streamlit dashboard (talks to the FastAPI service).

Run:  streamlit run dashboard/app.py        (set API_URL if the API is not on localhost:8000)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_ICON = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

st.set_page_config(page_title="RouteMind AI", page_icon="🚚", layout="wide")


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


st.title("🚚 RouteMind AI")
st.caption("Predictive Delivery Disruption & Recovery Platform")

try:
    health = api_get("/health")
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app --port 8000`.\n\n{e}")
    st.stop()

page = st.sidebar.radio("Page", ["📊 Overview", "🔮 Predict a delivery", "🩺 Model monitor"])
st.sidebar.markdown(f"**API:** `{API_URL}`  \n**Model:** `{health['model_version']}`")

# --------------------------------------------------------------- overview
if page.startswith("📊"):
    s = api_get("/monitoring/summary")
    dist = s["risk_distribution"]
    risky = dist.get("HIGH", 0) + dist.get("CRITICAL", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders scored", f"{s['n_predictions']:,}")
    c2.metric("High-risk orders", f"{risky:,}")
    c3.metric("Avg predicted delay", f"{s['avg_expected_delay_minutes'] or 0:.1f} min")
    c4.metric("Model status", s["model_status"].replace("_", " ").title())

    st.subheader("Delivery risk distribution")
    if dist:
        st.bar_chart(pd.Series({k: dist.get(k, 0) for k in RISK_ORDER}, name="orders"))
    else:
        st.info("No predictions yet. Use the Predict page or run `python -m scripts.simulate_predictions`.")

    st.subheader("Highest-risk deliveries")
    rows = api_get("/monitoring/recent", limit=15, risky_only=True)
    if rows:
        df = pd.DataFrame(rows)
        df["risk_level"] = df["risk_level"].map(lambda r: f"{RISK_ICON[r]} {r}")
        df["delay_probability"] = (df["delay_probability"] * 100).round(1).astype(str) + " %"
        st.dataframe(df.drop(columns=["timestamp"]), width="stretch", hide_index=True)
    else:
        st.write("No HIGH / CRITICAL deliveries yet.")

# --------------------------------------------------------------- predict
elif page.startswith("🔮"):
    st.subheader("Score a delivery")
    with st.form("predict"):
        a, b, c = st.columns(3)
        order_id = a.text_input("Order ID", "ORD-10231")
        distance = a.number_input("Distance (km)", 0.1, 200.0, 18.4)
        traffic = a.slider("Traffic level (0-10)", 0.0, 10.0, 8.0, 0.5)
        congestion = a.slider("Route congestion (0-10)", 0.0, 10.0, 6.0, 0.5)
        vehicle = b.selectbox("Vehicle", ["bike", "scooter", "car", "van"])
        zone = b.selectbox("Delivery zone", ["urban", "suburban", "rural"])
        hour = b.slider("Hour of day", 0, 23, 18)
        temp = b.number_input("Temperature (°C)", -10.0, 55.0, 31.0)
        exp = c.number_input("Driver experience (years)", 0.0, 45.0, 2.4)
        speed = c.number_input("Current speed (km/h)", 0.0, 120.0, 18.0)
        rain = c.checkbox("Raining", value=True)
        go = st.form_submit_button("Predict", type="primary")

    if go:
        payload = {
            "order_id": order_id, "distance_km": distance, "traffic_level": traffic,
            "temperature": temp, "rain": rain, "vehicle_type": vehicle, "delivery_zone": zone,
            "hour": hour, "driver_experience": exp, "current_speed": speed,
            "route_congestion": congestion,
        }
        try:
            res = api_post("/predict", payload)
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.text}")
            st.stop()

        st.markdown(f"### {RISK_ICON[res['risk_level']]} Risk: **{res['risk_level']}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of delay", f"{res['delay_probability'] * 100:.1f} %")
        m2.metric("Expected delay", f"{res['expected_delay_minutes']:.0f} min")
        m3.metric("Predicted class", res["predicted_class"].replace("_", " "))

        left, right = st.columns(2)
        with left:
            st.markdown("**Main risk factors**")
            if res["risk_factors"]:
                st.dataframe(pd.DataFrame(res["risk_factors"]), hide_index=True, width="stretch")
            else:
                st.write("No significant risk factors.")
        with right:
            st.markdown("**Recommended action**")
            for act in res["recommended_actions"]:
                st.write(f"→ {act}")

# --------------------------------------------------------------- monitor
else:
    s = api_get("/monitoring/summary")
    st.subheader("Model monitor")
    tm = s["training_metrics"]
    if tm:
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                            ("Recall", "recall"), ("F1 (macro)", "f1_macro")] ):
            col.metric(label, f"{tm[key] * 100:.2f} %")
        cols[4].metric("MAE", f"{tm['mae']:.2f} min")
        st.caption("Metrics measured on the held-out test set at training time.")

    d = s["drift"]
    st.markdown(f"**Data drift status:** `{d['status']}`  ·  **Model status:** `{s['model_status']}`")
    if d["features"]:
        st.metric("Features drifted", f"{d.get('drift_pct', 0):.0f} %")
        psi = pd.Series(d["features"], name="PSI").sort_values(ascending=False)
        st.bar_chart(psi)
        st.caption(f"PSI above {d['psi_threshold']} = live data no longer looks like the training data.")
    else:
        st.info(f"Drift needs at least {d.get('min_samples', 200)} scored deliveries "
                f"(have {d.get('n_samples', 0)}).")

    p = s["live_performance"]
    st.markdown("**Live accuracy (from /feedback)**")
    if p.get("available"):
        a, b = st.columns(2)
        a.metric("Live MAE", f"{p['live_mae']} min")
        b.metric("Live class accuracy", f"{p['live_accuracy'] * 100:.1f} %")
    else:
        st.write(f"Waiting for ground truth ({p.get('n', 0)} feedback rows so far).")

    st.markdown("**Serving stats**")
    x, y = st.columns(2)
    x.metric("Avg latency / record", f"{s['avg_latency_ms'] or 0} ms")
    y.metric("p95 latency / record", f"{s['p95_latency_ms'] or 0} ms")
