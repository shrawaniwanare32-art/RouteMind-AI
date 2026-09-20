"""Step 4 - train the two ML models (classifier + regressor)."""
import sys

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.components.data_transformation import TransformedData, build_model_pipeline
from src.config.configuration import AppConfig
from src.utils.exception import CustomException, ModelTrainingException
from src.utils.logger import get_logger

logger = get_logger("model_trainer")


class ModelTrainer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def train(self, data: TransformedData) -> tuple[Pipeline, Pipeline]:
        try:
            m = self.cfg["model"]
            logger.info("Model training started (%s training rows)", f"{len(data.X_train):,}")

            # Model 1 - Delay classification: ON_TIME / SLIGHT_DELAY / HIGH_DELAY
            clf = build_model_pipeline(HistGradientBoostingClassifier(**m["classifier"]))
            clf.fit(data.X_train, data.y_class_train)
            logger.info("Classifier trained")

            # Model 2 - Delay regression: expected delay in minutes
            reg = build_model_pipeline(HistGradientBoostingRegressor(**m["regressor"]))
            reg.fit(data.X_train, data.y_reg_train)
            logger.info("Regressor trained")
            return clf, reg
        except CustomException:
            raise
        except Exception as e:
            raise ModelTrainingException(e, sys) from e
