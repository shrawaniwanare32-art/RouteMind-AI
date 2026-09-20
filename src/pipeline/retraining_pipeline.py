"""Retraining pipeline: check drift / live accuracy -> retrain -> deploy only if better.

Run:  python -m src.pipeline.retraining_pipeline [--force]
"""
import argparse
import sys

from src.components.monitoring import MonitoringService
from src.config.configuration import AppConfig, load_config
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.exception import CustomException, ModelTrainingException
from src.utils.logger import get_logger

logger = get_logger("retraining_pipeline")


class RetrainingPipeline:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()

    def run(self, force: bool = False) -> dict:
        try:
            monitor = MonitoringService(self.cfg)
            drift = monitor.drift_report()
            perf = monitor.performance_report()
            triggers = []
            if force:
                triggers.append("forced")
            if drift["drift_detected"]:
                triggers.append(f"data drift on {drift['drifted_features']}")
            if perf.get("degraded"):
                triggers.append(f"live MAE degraded ({perf['live_mae']} min)")

            if not triggers:
                logger.info("Retraining not needed (drift status: %s)", drift["status"])
                return {"retrained": False, "reason": f"no trigger (drift status: {drift['status']})"}

            logger.info("Retraining triggered: %s", "; ".join(triggers))
            result = TrainingPipeline(self.cfg).run(include_new_data=True)
            return {"retrained": True, "triggers": triggers, **result}
        except CustomException:
            raise
        except Exception as e:
            raise ModelTrainingException(e, sys) from e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    res = RetrainingPipeline().run(force=ap.parse_args().force)
    if res["retrained"]:
        print(f"\nRetrained -> {res['version']} | promoted: {res['promoted']} | {res['reason']}")
    else:
        print(f"\nSkipped: {res['reason']}")
