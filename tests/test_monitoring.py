import numpy as np
import pandas as pd

from src.components.data_generator import generate_deliveries
from src.components.monitoring import MonitoringService, PredictionLogger, compute_psi
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.retraining_pipeline import RetrainingPipeline


def _reset_logs(cfg):
    for name in ("predictions.jsonl", "feedback.jsonl"):
        p = cfg.path("logs_dir") / name
        if p.exists():
            p.unlink()


def _simulate(cfg, n, drift, seed, feedback=False):
    plog = PredictionLogger(cfg)
    pipe = PredictionPipeline(cfg, prediction_logger=plog)
    df = generate_deliveries(n, seed=seed, drift=drift)
    df["order_id"] = [f"T{seed}-{i}" for i in range(n)]
    pipe.predict(df.drop(columns=["delay_minutes"]).to_dict("records"))
    if feedback:
        for oid, d in zip(df["order_id"], df["delay_minutes"]):
            plog.log_feedback(oid, d)


def test_psi_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert compute_psi(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert compute_psi(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25
    cat_a, cat_b = pd.Series(["x"] * 900 + ["y"] * 100), pd.Series(["x"] * 300 + ["y"] * 700)
    assert compute_psi(cat_a, cat_b) > 0.25


def test_insufficient_data_then_stable_then_drift(trained, cfg):
    _reset_logs(cfg)
    mon = MonitoringService(cfg)
    assert mon.drift_report()["status"] == "INSUFFICIENT_DATA"

    _simulate(cfg, 600, drift=False, seed=11)
    stable = mon.drift_report()
    assert stable["status"] == "STABLE", stable

    _reset_logs(cfg)
    _simulate(cfg, 600, drift=True, seed=12)
    drifted = mon.drift_report()
    assert drifted["status"] == "DRIFT_DETECTED"
    assert "traffic_level" in drifted["drifted_features"]
    assert mon.summary()["model_status"] == "RETRAIN_RECOMMENDED"


def test_live_performance_from_feedback(trained, cfg):
    _reset_logs(cfg)
    _simulate(cfg, 300, drift=False, seed=21, feedback=True)
    perf = MonitoringService(cfg).performance_report()
    assert perf["available"] and perf["n"] == 300
    assert perf["live_mae"] < 6 and perf["live_accuracy"] > 0.7
    assert perf["degraded"] is False


def test_retraining_skipped_when_stable_and_runs_when_forced(trained, cfg):
    _reset_logs(cfg)
    _simulate(cfg, 400, drift=False, seed=31)
    assert RetrainingPipeline(cfg).run()["retrained"] is False
    forced = RetrainingPipeline(cfg).run(force=True)
    assert forced["retrained"] is True and "forced" in forced["triggers"]
