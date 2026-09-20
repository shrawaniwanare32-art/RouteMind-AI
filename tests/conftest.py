"""Tests run in a throw-away project home so they never touch your real data/models."""
import os
import tempfile

_TMP_HOME = tempfile.mkdtemp(prefix="routemind_test_")
os.environ["ROUTEMIND_HOME"] = _TMP_HOME          # must be set BEFORE importing src
os.environ["ROUTEMIND_N_SAMPLES"] = "3000"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    from src.pipeline.training_pipeline import TrainingPipeline
    return TrainingPipeline().run()


@pytest.fixture()
def cfg():
    from src.config.configuration import load_config
    return load_config()


@pytest.fixture()
def easy_order():
    return dict(distance_km=3, traffic_level=2, rain=False, vehicle_type="car",
                delivery_zone="suburban", hour=14, driver_experience=8, current_speed=34)


@pytest.fixture()
def hard_order():
    return dict(distance_km=18.4, traffic_level=8, temperature=31, rain=True, vehicle_type="bike",
                delivery_zone="urban", hour=18, driver_experience=2.4, current_speed=18)
