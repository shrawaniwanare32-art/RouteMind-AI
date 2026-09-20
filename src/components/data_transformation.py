"""Step 3 - feature engineering, target creation, deterministic train/test split.

The feature engineering + preprocessing lives INSIDE the sklearn Pipeline that is
saved with the model, so training and serving always use identical logic.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config.configuration import AppConfig
from src.utils.exception import CustomException, DataTransformationException
from src.utils.logger import get_logger
from src.utils.schema import (
    CATEGORICAL_FEATURES,
    MODEL_NUMERIC,
    PEAK_HOURS,
    RAW_FEATURES,
    TARGET_COL,
)

logger = get_logger("data_transformation")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Traffic / time / route features. Module-level so the fitted pipeline can be pickled."""
    df = df.copy()
    speed = df["current_speed"].clip(lower=3)
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["eta_travel_min"] = df["distance_km"] / speed * 60
    df["traffic_x_distance"] = df["traffic_level"] * df["distance_km"]
    df["rain_x_traffic"] = df["rain"] * df["traffic_level"]
    return df


def build_model_pipeline(estimator) -> Pipeline:
    """features -> impute/encode -> estimator"""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric, MODEL_NUMERIC), ("cat", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def label_delay(minutes, on_time_max: float, slight_max: float) -> np.ndarray:
    minutes = np.asarray(minutes, dtype=float)
    return np.where(
        minutes <= on_time_max,
        "ON_TIME",
        np.where(minutes <= slight_max, "SLIGHT_DELAY", "HIGH_DELAY"),
    )


def split_mask_by_hash(df: pd.DataFrame, test_size: float) -> np.ndarray:
    """Deterministic train/test assignment (True = test).

    A delivery is assigned by hashing its order_id, so it lands in the SAME split on every
    retraining run. That keeps the test set free of rows any earlier model was trained on,
    which is what makes the champion-vs-challenger comparison fair. (A fresh random split
    on every run would let the production model be scored on its own training data.)
    """
    if "order_id" in df.columns:
        keys = df["order_id"].astype(str)
    else:  # no id column -> hash the feature values instead
        keys = df[RAW_FEATURES].astype(str).agg("|".join, axis=1)
    buckets = pd.util.hash_pandas_object(keys, index=False).to_numpy() % 100
    return buckets < int(round(test_size * 100))


@dataclass
class TransformedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_class_train: np.ndarray
    y_class_test: np.ndarray
    y_reg_train: np.ndarray
    y_reg_test: np.ndarray
    reference: pd.DataFrame  # sample of training inputs, used later as the drift baseline


class DataTransformation:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self, validated_path: Path) -> TransformedData:
        try:
            logger.info("Feature engineering started")
            df = pd.read_csv(validated_path)
            t = self.cfg["target"]
            d = self.cfg["data"]

            y_reg = df[TARGET_COL].to_numpy(dtype=float)
            y_class = label_delay(y_reg, t["on_time_max_minutes"], t["slight_delay_max_minutes"])
            X = df[RAW_FEATURES]

            is_test = split_mask_by_hash(df, d["test_size"])
            X_tr, X_te = X[~is_test], X[is_test]
            yc_tr, yc_te = y_class[~is_test], y_class[is_test]
            yr_tr, yr_te = y_reg[~is_test], y_reg[is_test]

            processed = self.cfg.ensure_dir("processed_dir")
            X_tr.assign(delay_minutes=yr_tr).to_csv(processed / "train.csv", index=False)
            X_te.assign(delay_minutes=yr_te).to_csv(processed / "test.csv", index=False)

            n_ref = min(d["reference_sample_size"], len(X_tr))
            reference = X_tr.sample(n_ref, random_state=d["random_state"])

            dist = pd.Series(y_class).value_counts(normalize=True).round(3).to_dict()
            logger.info("Feature engineering completed | class balance: %s", dist)
            return TransformedData(X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, reference)
        except CustomException:
            raise
        except Exception as e:
            raise DataTransformationException(e, sys) from e
