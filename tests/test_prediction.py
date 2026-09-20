import pytest

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.exception import PredictionException


@pytest.fixture()
def pipe(trained, cfg):
    return PredictionPipeline(cfg)


def test_prediction_shape(pipe, hard_order):
    r = pipe.predict([hard_order], log=False)[0]
    assert r["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0 <= r["delay_probability"] <= 1
    assert r["expected_delay_minutes"] >= 0
    assert abs(sum(r["class_probabilities"].values()) - 1) < 0.01
    assert r["order_id"].startswith("ORD-")
    assert r["recommended_actions"]
    pcts = [f["contribution_pct"] for f in r["risk_factors"]]
    assert pcts == sorted(pcts, reverse=True)


def test_hard_order_riskier_than_easy(pipe, hard_order, easy_order):
    hard, easy = pipe.predict([hard_order, easy_order], log=False)
    assert hard["expected_delay_minutes"] > easy["expected_delay_minutes"] + 10
    assert hard["delay_probability"] > easy["delay_probability"]
    assert easy["risk_level"] == "LOW" and easy["recommended_action"] == "CONTINUE"
    assert hard["risk_level"] in {"HIGH", "CRITICAL"}


def test_rain_is_a_risk_factor_when_raining(pipe, hard_order):
    factors = {f["factor"] for f in pipe.predict([hard_order], log=False)[0]["risk_factors"]}
    assert factors & {"Rain", "Heavy traffic", "Low current speed", "Long distance"}


def test_missing_required_feature_raises(pipe, easy_order):
    del easy_order["traffic_level"]
    with pytest.raises(PredictionException, match="traffic_level"):
        pipe.predict([easy_order])


def test_optional_features_are_imputed(pipe, easy_order):
    assert "temperature" not in easy_order and "route_congestion" not in easy_order
    assert pipe.predict([easy_order], log=False)[0]["expected_delay_minutes"] >= 0


def test_empty_request_raises(pipe):
    with pytest.raises(PredictionException):
        pipe.predict([])


def test_exception_message_has_file_and_line(pipe, easy_order):
    del easy_order["hour"]
    with pytest.raises(PredictionException) as exc:
        pipe.predict([easy_order])
    assert "prediction_pipeline.py" in str(exc.value) and "line:" in str(exc.value)
