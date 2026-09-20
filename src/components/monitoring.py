"""Prediction logging, data-drift (PSI) and live-performance monitoring."""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.components.data_transformation import label_delay
from src.config.configuration import AppConfig
from src.models.model_loader import ModelRegistry
from src.utils.exception import CustomException, MonitoringException
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger
from src.utils.schema import RAW_FEATURES

logger = get_logger("monitoring")
_EPS = 1e-4


# ---------------------------------------------------------------- PSI helpers
def _psi_from_props(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    ref_p = np.clip(ref_p, _EPS, None)
    cur_p = np.clip(cur_p, _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return psi_categorical(ref, cur)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(ref, edges)[0] / len(ref)
    cur_p = np.histogram(cur, edges)[0] / len(cur)
    return _psi_from_props(ref_p, cur_p)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_p = np.array([(ref == c).mean() for c in cats])
    cur_p = np.array([(cur == c).mean() for c in cats])
    return _psi_from_props(ref_p, cur_p)


def compute_psi(ref: pd.Series, cur: pd.Series) -> float:
    return psi_categorical(ref, cur) if ref.nunique() <= 10 else psi_numeric(ref, cur)


# --------------------------------------------------------------- logging
class PredictionLogger:
    """Append-only JSONL logs. In the cloud, point these at a DB / S3 / Blob instead."""

    def __init__(self, cfg: AppConfig):
        logs = cfg.ensure_dir("logs_dir")
        self.predictions_path = logs / "predictions.jsonl"
        self.feedback_path = logs / "feedback.jsonl"
        self._lock = threading.Lock()

    def _append(self, path: Path, rows: list[dict]) -> None:
        with self._lock, open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def log_predictions(self, rows: list[dict]) -> None:
        stamp = utc_now_iso()
        self._append(self.predictions_path, [{"timestamp": stamp, **r} for r in rows])

    def log_feedback(self, order_id: str, actual_delay_minutes: float) -> None:
        self._append(
            self.feedback_path,
            [{"timestamp": utc_now_iso(), "order_id": order_id,
              "actual_delay_minutes": float(actual_delay_minutes)}],
        )


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False)


# ---------------------------------------------------------------- service
class MonitoringService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mcfg = cfg["monitoring"]
        self.registry = ModelRegistry(cfg)
        self.logger_ = PredictionLogger(cfg)

    def predictions(self) -> pd.DataFrame:
        return _read_jsonl(self.logger_.predictions_path)

    def recent_predictions(self, limit: int = 50, risky_only: bool = False) -> list[dict]:
        df = self.predictions()
        if df.empty:
            return []
        if risky_only:
            df = df[df["risk_level"].isin(["HIGH", "CRITICAL"])]
        df = df.sort_values("delay_probability", ascending=False) if risky_only else df.iloc[::-1]
        cols = ["timestamp", "order_id", "risk_level", "delay_probability",
                "expected_delay_minutes", "predicted_class"]
        return json.loads(df.head(limit)[cols].to_json(orient="records"))

    def drift_report(self) -> dict:
        try:
            ref_path = self.cfg.path("reference_dir") / "reference.csv"
            if not ref_path.exists():
                return {"status": "NO_REFERENCE", "drift_detected": False, "features": {}}
            cur = self.predictions()
            n = len(cur)
            if n < self.mcfg["min_samples"]:
                return {"status": "INSUFFICIENT_DATA", "drift_detected": False,
                        "n_samples": n, "min_samples": self.mcfg["min_samples"], "features": {}}
            cur = cur.tail(self.mcfg["window_size"])
            ref = pd.read_csv(ref_path)

            psi = {f: round(compute_psi(ref[f], cur[f]), 4)
                   for f in RAW_FEATURES if f in cur.columns and cur[f].notna().any()}
            drifted = [f for f, v in psi.items() if v > self.mcfg["psi_threshold"]]
            detected = len(drifted) >= self.mcfg["min_drifted_features"]
            return {
                "status": "DRIFT_DETECTED" if detected else "STABLE",
                "drift_detected": detected,
                "n_samples": int(len(cur)),
                "drift_pct": round(100 * len(drifted) / max(len(psi), 1), 1),
                "drifted_features": drifted,
                "psi_threshold": self.mcfg["psi_threshold"],
                "features": psi,
            }
        except CustomException:
            raise
        except Exception as e:
            raise MonitoringException(e, sys) from e

    def performance_report(self) -> dict:
        """Live accuracy - only possible once real outcomes come back via /feedback."""
        preds, fb = self.predictions(), _read_jsonl(self.logger_.feedback_path)
        if preds.empty or fb.empty:
            return {"available": False, "n": 0}
        joined = preds.drop_duplicates("order_id", keep="last").merge(
            fb.drop_duplicates("order_id", keep="last")[["order_id", "actual_delay_minutes"]],
            on="order_id",
        )
        n = len(joined)
        if n < self.mcfg["feedback_min_samples"]:
            return {"available": False, "n": n, "needed": self.mcfg["feedback_min_samples"]}

        t = self.cfg["target"]
        actual_cls = label_delay(joined["actual_delay_minutes"], t["on_time_max_minutes"],
                                 t["slight_delay_max_minutes"])
        mae = float((joined["expected_delay_minutes"] - joined["actual_delay_minutes"]).abs().mean())
        acc = float((actual_cls == joined["predicted_class"].to_numpy()).mean())

        degraded = False
        version = self.registry.latest_version()
        if version:
            train_mae = self.registry.load(version).metadata["metrics"]["mae"]
            degraded = mae > train_mae * self.mcfg["mae_degradation_ratio"]
        return {"available": True, "n": n, "live_mae": round(mae, 3),
                "live_accuracy": round(acc, 4), "degraded": degraded}

    def summary(self) -> dict:
        preds = self.predictions()
        version = self.registry.latest_version()
        meta = self.registry.load(version).metadata if version else {}
        drift = self.drift_report()
        perf = self.performance_report()

        out = {
            "model_version": version,
            "training_metrics": meta.get("metrics", {}),
            "n_predictions": int(len(preds)),
            "risk_distribution": {},
            "avg_expected_delay_minutes": None,
            "avg_latency_ms": None,
            "p95_latency_ms": None,
            "drift": drift,
            "live_performance": perf,
        }
        if not preds.empty:
            out["risk_distribution"] = preds["risk_level"].value_counts().to_dict()
            out["avg_expected_delay_minutes"] = round(float(preds["expected_delay_minutes"].mean()), 2)
            if "latency_ms" in preds:
                out["avg_latency_ms"] = round(float(preds["latency_ms"].mean()), 2)
                out["p95_latency_ms"] = round(float(preds["latency_ms"].quantile(0.95)), 2)

        if drift["drift_detected"] or perf.get("degraded"):
            out["model_status"] = "RETRAIN_RECOMMENDED"
        elif drift["status"] == "INSUFFICIENT_DATA":
            out["model_status"] = "INSUFFICIENT_DATA"
        else:
            out["model_status"] = "HEALTHY"
        return out
