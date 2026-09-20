from src.models.model_loader import ModelRegistry
from src.pipeline.training_pipeline import TrainingPipeline


def test_training_produces_good_model_and_promotes(trained, cfg):
    assert trained["promoted"] is True
    m = trained["metrics"]
    assert m["accuracy"] > 0.70
    assert m["roc_auc_delay"] > 0.85
    assert m["r2"] > 0.7
    reg = ModelRegistry(cfg)
    assert reg.latest_version() is not None          # (other tests may have promoted a newer one)
    assert trained["version"] in {v["version"] for v in reg.list_versions()}


def test_second_run_only_promotes_if_not_worse(trained, cfg):
    res = TrainingPipeline(cfg).run()          # same data, same seed -> equal quality
    assert res["production_metrics_on_same_test"] is not None
    assert res["promoted"] is True             # identical scores pass the >= rule
    versions = ModelRegistry(cfg).list_versions()
    assert sum(v["is_production"] for v in versions) == 1


def test_bad_candidate_is_rejected(cfg, trained):
    from src.components.model_evaluation import ModelEvaluation
    ev = ModelEvaluation(cfg)
    prod = {"f1_macro": 0.90, "mae": 2.0}
    assert ev.should_promote({"f1_macro": 0.80, "mae": 2.0}, prod)[0] is False
    assert ev.should_promote({"f1_macro": 0.91, "mae": 2.5}, prod)[0] is False   # MAE too much worse
    assert ev.should_promote({"f1_macro": 0.91, "mae": 2.0}, prod)[0] is True


def test_drift_baseline_only_changes_when_a_model_is_promoted(trained, cfg):
    import pandas as pd

    from src.components.data_generator import generate_deliveries

    ref_path = cfg.path("reference_dir") / "reference.csv"
    before = ref_path.read_text()
    reg = ModelRegistry(cfg)
    prod = reg.load()
    other_ref = generate_deliveries(50, seed=77)[["distance_km"]]

    v = reg.save_candidate(prod.classifier, prod.regressor, prod.metadata["metrics"],
                           reference=pd.DataFrame(other_ref))
    assert ref_path.read_text() == before          # saved with the version, NOT activated
    reg.promote(v)
    assert ref_path.read_text() != before          # activated together with the model
    reg.promote(prod.version)                      # restore for the other tests
    assert ref_path.read_text() == before
