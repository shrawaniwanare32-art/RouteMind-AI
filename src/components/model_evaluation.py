"""Step 5 - metrics + the champion/challenger promotion rule."""
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.components.data_transformation import TransformedData
from src.config.configuration import AppConfig
from src.utils.exception import CustomException, ModelTrainingException
from src.utils.logger import get_logger
from src.utils.schema import CLASS_LABELS

logger = get_logger("model_evaluation")


class ModelEvaluation:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def evaluate(self, clf, reg, data: TransformedData) -> dict:
        try:
            X, yc, yr = data.X_test, data.y_class_test, data.y_reg_test
            pred = clf.predict(X)
            proba = clf.predict_proba(X)
            classes = list(clf.classes_)
            p_delay = 1.0 - proba[:, classes.index("ON_TIME")]
            is_delayed = (yc != "ON_TIME").astype(int)

            reg_pred = np.clip(reg.predict(X), 0, None)
            metrics = {
                "accuracy": accuracy_score(yc, pred),
                "precision": precision_score(yc, pred, average="macro", zero_division=0),
                "recall": recall_score(yc, pred, average="macro", zero_division=0),
                "f1_macro": f1_score(yc, pred, average="macro", zero_division=0),
                "roc_auc_delay": roc_auc_score(is_delayed, p_delay),
                "mae": mean_absolute_error(yr, reg_pred),
                "rmse": float(np.sqrt(mean_squared_error(yr, reg_pred))),
                "r2": r2_score(yr, reg_pred),
                "n_test": int(len(yc)),
                "confusion_matrix": {
                    "labels": CLASS_LABELS,
                    "matrix": confusion_matrix(yc, pred, labels=CLASS_LABELS).tolist(),
                },
            }
            return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in metrics.items()}
        except CustomException:
            raise
        except Exception as e:
            raise ModelTrainingException(e, sys) from e

    def should_promote(self, candidate: dict, production: dict | None) -> tuple[bool, str]:
        """Deploy only if better: macro-F1 must not drop and MAE must not get much worse."""
        if production is None:
            return True, "no production model yet"
        p = self.cfg["promotion"]
        f1_ok = candidate["f1_macro"] >= production["f1_macro"] + p["min_f1_improvement"]
        mae_ok = candidate["mae"] <= production["mae"] * (1 + p["max_mae_increase"])
        if f1_ok and mae_ok:
            return True, (
                f"candidate F1 {candidate['f1_macro']} >= production {production['f1_macro']} "
                f"and MAE {candidate['mae']} within tolerance of {production['mae']}"
            )
        return False, (
            f"candidate not better (F1 {candidate['f1_macro']} vs {production['f1_macro']}, "
            f"MAE {candidate['mae']} vs {production['mae']})"
        )
