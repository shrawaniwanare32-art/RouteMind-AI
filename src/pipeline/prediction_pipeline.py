"""Prediction pipeline: raw request(s) -> risk, probability, expected delay,
risk factors and a recommended recovery action."""
import sys
import time
import uuid

import numpy as np
import pandas as pd

from src.components.monitoring import PredictionLogger
from src.components.recovery_engine import recommend, risk_level
from src.config.configuration import AppConfig, load_config
from src.models.model_loader import ModelRegistry
from src.utils.exception import CustomException, PredictionException
from src.utils.logger import get_logger
from src.utils.schema import NUMERIC_INPUTS, RAW_FEATURES, REQUIRED_INPUTS

logger = get_logger("prediction_pipeline")

# "What if this factor were normal?" baselines used to explain each prediction.
FACTOR_BASELINES = {
    "Heavy traffic": {"traffic_level": 3.0},
    "Rain": {"rain": 0},
    "Peak hour": {"hour": 14},
    "Route congestion": {"route_congestion": 2.0},
    "Long distance": {"distance_km": 5.0},
    "Low current speed": {"current_speed": 30.0},
    "Inexperienced driver": {"driver_experience": 5.0},
}


class PredictionPipeline:
    def __init__(self, cfg: AppConfig | None = None, prediction_logger: PredictionLogger | None = None):
        self.cfg = cfg or load_config()
        self.registry = ModelRegistry(self.cfg)
        self.prediction_logger = prediction_logger
        self.reload()

    def reload(self) -> str:
        """(Re)load the production model - call after a retrain promoted a new version."""
        self.bundle = self.registry.load()
        logger.info("Loaded model %s", self.bundle.version)
        return self.bundle.version

    @property
    def version(self) -> str:
        return self.bundle.version

    # ----------------------------------------------------------- internals
    def _to_frame(self, records: list[dict]) -> pd.DataFrame:
        if not records:
            raise PredictionException("No records supplied")
        df = pd.DataFrame(records)
        for col in REQUIRED_INPUTS:
            if col not in df.columns or df[col].isna().any():
                raise PredictionException(f"Required feature '{col}' is missing")
        for col in RAW_FEATURES:
            if col not in df.columns:
                df[col] = np.nan  # optional inputs get imputed by the model pipeline
        try:
            for col in NUMERIC_INPUTS:
                df[col] = pd.to_numeric(df[col].astype(float) if col == "rain" else df[col])
        except (TypeError, ValueError) as e:
            raise PredictionException(f"Invalid numeric value: {e}") from e
        for col in ("vehicle_type", "delivery_zone"):
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df[RAW_FEATURES]

    def _explain(self, X: pd.DataFrame, expected: np.ndarray) -> list[list[dict]]:
        """Counterfactual attribution: how many minutes disappear if a factor were normal?"""
        n, names = len(X), list(FACTOR_BASELINES)
        blocks = []
        for name in names:
            alt = X.copy()
            for col, val in FACTOR_BASELINES[name].items():
                alt[col] = val
            blocks.append(alt)
        alt_pred = np.clip(self.bundle.regressor.predict(pd.concat(blocks, ignore_index=True)), 0, None)
        alt_pred = alt_pred.reshape(len(names), n)

        results = []
        for i in range(n):
            impact = {name: max(0.0, float(expected[i] - alt_pred[k, i])) for k, name in enumerate(names)}
            total = sum(impact.values())
            factors = []
            if total > 0:
                for name, mins in sorted(impact.items(), key=lambda kv: kv[1], reverse=True)[:4]:
                    if mins >= 0.5:
                        factors.append({"factor": name, "impact_minutes": round(mins, 1),
                                        "contribution_pct": round(100 * mins / total, 1)})
            results.append(factors)
        return results

    # ------------------------------------------------------------- public
    def predict(self, records: list[dict], log: bool = True) -> list[dict]:
        try:
            start = time.perf_counter()
            df = self._to_frame(records)
            b = self.bundle

            proba = b.classifier.predict_proba(df)
            classes = list(b.classifier.classes_)
            p_delay = 1.0 - proba[:, classes.index("ON_TIME")]
            expected = np.clip(b.regressor.predict(df), 0, None)
            explanations = self._explain(df, expected)
            latency_ms = (time.perf_counter() - start) * 1000 / len(df)

            out, log_rows = [], []
            for i, rec in enumerate(records):
                lvl = risk_level(float(p_delay[i]), float(expected[i]), self.cfg["risk"])
                code, actions = recommend(lvl, explanations[i], df.iloc[i]["vehicle_type"])
                order_id = rec.get("order_id") or f"ORD-{uuid.uuid4().hex[:8].upper()}"
                result = {
                    "order_id": order_id,
                    "risk_level": lvl,
                    "delay_probability": round(float(p_delay[i]), 4),
                    "predicted_class": classes[int(np.argmax(proba[i]))],
                    "class_probabilities": {c: round(float(proba[i, k]), 4) for k, c in enumerate(classes)},
                    "expected_delay_minutes": round(float(expected[i]), 1),
                    "risk_factors": explanations[i],
                    "recommended_action": code,
                    "recommended_actions": actions,
                    "model_version": b.version,
                }
                out.append(result)
                log_rows.append({
                    "order_id": order_id,
                    **{c: (None if pd.isna(df.iloc[i][c]) else df.iloc[i][c].item()
                           if hasattr(df.iloc[i][c], "item") else df.iloc[i][c]) for c in RAW_FEATURES},
                    "risk_level": lvl,
                    "delay_probability": result["delay_probability"],
                    "expected_delay_minutes": result["expected_delay_minutes"],
                    "predicted_class": result["predicted_class"],
                    "model_version": b.version,
                    "latency_ms": round(latency_ms, 2),
                })
            if log and self.prediction_logger is not None:
                self.prediction_logger.log_predictions(log_rows)
            logger.info("Prediction request handled: %d record(s), %.1f ms/record", len(out), latency_ms)
            return out
        except CustomException:
            raise
        except Exception as e:
            raise PredictionException(e, sys) from e
