import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.components.data_generator import generate_deliveries
from src.components.data_transformation import (
    add_features,
    build_model_pipeline,
    label_delay,
    split_mask_by_hash,
)
from src.utils.schema import ENGINEERED_FEATURES, RAW_FEATURES


def test_add_features_creates_columns_and_peak_flag():
    df = generate_deliveries(50, seed=3)[RAW_FEATURES]
    out = add_features(df)
    assert set(ENGINEERED_FEATURES).issubset(out.columns)
    assert out.loc[df["hour"] == 18, "is_peak_hour"].eq(1).all()
    assert out.loc[df["hour"] == 3, "is_peak_hour"].eq(0).all()


def test_label_delay_thresholds():
    labels = label_delay([0, 5, 5.1, 20, 20.1, 90], 5, 20)
    assert list(labels) == ["ON_TIME", "ON_TIME", "SLIGHT_DELAY", "SLIGHT_DELAY", "HIGH_DELAY", "HIGH_DELAY"]


def test_pipeline_handles_missing_values():
    df = generate_deliveries(400, seed=4, missing_rate=0.1)
    X, y = df[RAW_FEATURES], df["delay_minutes"]
    pipe = build_model_pipeline(HistGradientBoostingRegressor(max_iter=20)).fit(X, y)
    assert not np.isnan(pipe.predict(X)).any()
    assert isinstance(pipe.predict(X.head(1)), np.ndarray)


def test_training_reference_and_split_saved(trained, cfg):
    assert (cfg.path("reference_dir") / "reference.csv").exists()
    train = pd.read_csv(cfg.path("processed_dir") / "train.csv")
    test = pd.read_csv(cfg.path("processed_dir") / "test.csv")
    assert len(train) + len(test) == 3000


def test_split_is_stable_across_retrains():
    """An order must land in the same split even when more data is added later,
    otherwise the production model could be scored on rows it was trained on."""
    big = generate_deliveries(3000, seed=8)
    small = big.iloc[:2000]
    test_big = set(big.loc[split_mask_by_hash(big, 0.2), "order_id"])
    test_small = set(small.loc[split_mask_by_hash(small, 0.2), "order_id"])
    assert test_small == test_big & set(small["order_id"])
    assert 0.15 < len(test_big) / len(big) < 0.25
