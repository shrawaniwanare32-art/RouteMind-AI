import pandas as pd
import pytest

from src.components.data_generator import generate_deliveries
from src.components.data_validation import DataValidation
from src.utils.exception import DataValidationException


def _write(cfg, df, name="in.csv"):
    p = cfg.ensure_dir("processed_dir") / name
    df.to_csv(p, index=False)
    return p


def test_validation_drops_impossible_rows_and_duplicates(cfg):
    df = generate_deliveries(500, seed=1)
    df.loc[0, "distance_km"] = -5            # impossible
    df.loc[1, "vehicle_type"] = "spaceship"  # unknown category -> imputed, row kept
    df.loc[2, "delay_minutes"] = None        # no target -> removed
    df = pd.concat([df, df.iloc[[10]]])      # duplicate order_id
    out, report = DataValidation(cfg).run(_write(cfg, df))
    clean = pd.read_csv(out)
    assert report["rows_in"] == 501
    assert report["duplicates_removed"] == 1
    assert report["out_of_range_rows_removed"] == 1
    assert report["invalid_target_rows_removed"] == 1
    assert (clean["distance_km"] > 0).all()
    assert len(clean) == 498


def test_validation_fails_on_missing_column(cfg):
    df = generate_deliveries(500, seed=1).drop(columns=["traffic_level"])
    with pytest.raises(DataValidationException, match="traffic_level"):
        DataValidation(cfg).run(_write(cfg, df))


def test_validation_fails_when_too_many_missing(cfg):
    df = generate_deliveries(500, seed=1)
    df.loc[:200, "current_speed"] = None
    with pytest.raises(DataValidationException, match="Too many missing"):
        DataValidation(cfg).run(_write(cfg, df))
