"""Training pipeline: Ingestion -> Validation -> Transformation -> Training ->
Evaluation -> Model Registry -> (promote only if better).

Run:  python -m src.pipeline.training_pipeline
"""
import argparse
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import AppConfig, load_config
from src.models.model_loader import ModelRegistry
from src.utils.exception import CustomException, ModelTrainingException
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")


class TrainingPipeline:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()

    def run(self, include_new_data: bool = False, force_promote: bool = False) -> dict:
        try:
            logger.info("=== Training pipeline started ===")
            ingested = DataIngestion(self.cfg).run(include_new_data=include_new_data)
            validated, val_report = DataValidation(self.cfg).run(ingested)
            data = DataTransformation(self.cfg).run(validated)
            clf, reg = ModelTrainer(self.cfg).train(data)

            evaluator = ModelEvaluation(self.cfg)
            metrics = evaluator.evaluate(clf, reg, data)
            logger.info("Model accuracy: %.2f%% | F1(macro): %.4f | MAE: %.2f min",
                        metrics["accuracy"] * 100, metrics["f1_macro"], metrics["mae"])

            registry = ModelRegistry(self.cfg)
            version = registry.save_candidate(
                clf, reg, metrics,
                extra={"rows_trained": int(len(data.X_train)), "validation": val_report,
                       "included_new_data": include_new_data},
                reference=data.reference,
            )
            logger.info("Candidate model saved: %s", version)

            # champion / challenger: score the current production model on the SAME test set
            production_metrics = None
            prod_version = registry.latest_version()
            if prod_version:
                try:
                    prod = registry.load(prod_version)
                    production_metrics = evaluator.evaluate(prod.classifier, prod.regressor, data)
                except Exception as e:  # e.g. old pickle no longer loadable
                    logger.warning("Could not evaluate production model (%s) - promoting candidate", e)

            promote, reason = evaluator.should_promote(metrics, production_metrics)
            if force_promote and not promote:
                promote, reason = True, "forced promotion"
            if promote:
                registry.promote(version)
                logger.info("Model promoted to production: %s (%s)", version, reason)
            else:
                logger.info("Candidate NOT promoted: %s", reason)

            summary = {"version": version, "promoted": promote, "reason": reason,
                       "metrics": metrics, "production_metrics_on_same_test": production_metrics}
            save_json(summary, self.cfg.ensure_dir("metrics_dir") / "last_training_run.json")
            logger.info("=== Training pipeline finished ===")
            return summary
        except CustomException:
            raise
        except Exception as e:
            raise ModelTrainingException(e, sys) from e


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train RouteMind models")
    ap.add_argument("--include-new-data", action="store_true",
                    help="also use data/external/new_deliveries.csv")
    ap.add_argument("--force-promote", action="store_true")
    a = ap.parse_args()
    res = TrainingPipeline().run(include_new_data=a.include_new_data, force_promote=a.force_promote)
    print(f"\nVersion: {res['version']} | promoted: {res['promoted']} | reason: {res['reason']}")
    m = res["metrics"]
    print(f"Accuracy {m['accuracy']:.2%} | Precision {m['precision']:.2%} | Recall {m['recall']:.2%} "
          f"| F1 {m['f1_macro']:.2%} | MAE {m['mae']} min | R2 {m['r2']}")
