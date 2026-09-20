import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.utils.schema import RAW_FEATURES, TARGET_COL


def test_ingestion_generates_and_saves_dataset(cfg):
    path = DataIngestion(cfg).run()
    df = pd.read_csv(path)
    assert len(df) == 3000
    assert set(RAW_FEATURES + [TARGET_COL]).issubset(df.columns)
    assert cfg.path("raw_data").exists()


def test_ingestion_merges_new_data(cfg):
    from src.components.data_generator import generate_deliveries
    DataIngestion(cfg).run()
    new = generate_deliveries(100, seed=5)
    new["order_id"] = [f"NEW-{i}" for i in range(100)]
    cfg.path("new_data").parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(cfg.path("new_data"), index=False)
    try:
        df = pd.read_csv(DataIngestion(cfg).run(include_new_data=True))
        assert len(df) == 3100
    finally:
        cfg.path("new_data").unlink()
