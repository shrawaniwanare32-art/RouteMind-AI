"""Fill the prediction log with traffic so the dashboard / drift monitor have data.

python -m scripts.simulate_predictions --n 600                  # normal traffic
python -m scripts.simulate_predictions --n 600 --drift          # drifted traffic
python -m scripts.simulate_predictions --n 600 --with-feedback  # also report real delays
"""
import argparse

from src.components.data_generator import generate_deliveries
from src.components.monitoring import PredictionLogger
from src.config.configuration import load_config
from src.pipeline.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--with-feedback", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    plog = PredictionLogger(cfg)
    pipe = PredictionPipeline(cfg, prediction_logger=plog)

    df = generate_deliveries(a.n, seed=a.seed, drift=a.drift)
    df["order_id"] = [f"SIM-{a.seed}-{i}" for i in range(len(df))]
    feats = df.drop(columns=["delay_minutes"])
    for start in range(0, len(feats), 200):
        pipe.predict(feats.iloc[start:start + 200].to_dict("records"))
    if a.with_feedback:
        for oid, d in zip(df["order_id"], df["delay_minutes"]):
            plog.log_feedback(oid, d)
    print(f"Logged {len(df)} predictions (drift={a.drift}, feedback={a.with_feedback})")
