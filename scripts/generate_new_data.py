"""Write fresh labelled deliveries to data/external/new_deliveries.csv
(the retraining pipeline merges this with the original dataset).

python -m scripts.generate_new_data --n 6000 --drift
"""
import argparse

from src.components.data_generator import generate_deliveries
from src.config.configuration import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--drift", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    out = cfg.path("new_data")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_deliveries(a.n, seed=a.seed, drift=a.drift, missing_rate=0.01)
    df["order_id"] = [f"NEW-{a.seed}-{i}" for i in range(len(df))]
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
