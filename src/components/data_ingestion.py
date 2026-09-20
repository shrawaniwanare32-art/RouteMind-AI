"""Step 1 - load raw data (and optional new data) into data/processed/ingested.csv."""
import sys
from pathlib import Path

import pandas as pd

from src.components.data_generator import generate_deliveries
from src.config.configuration import AppConfig
from src.utils.exception import CustomException, DataIngestionException
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")


class DataIngestion:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self, include_new_data: bool = False) -> Path:
        try:
            logger.info("Data ingestion started")
            raw_path = self.cfg.path("raw_data")
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            if not raw_path.exists():
                if not self.cfg["data"]["auto_generate"]:
                    raise FileNotFoundError(f"Raw data not found: {raw_path}")
                d = self.cfg["data"]
                logger.info("No raw data found - generating %s synthetic deliveries", d["n_samples"])
                generate_deliveries(
                    d["n_samples"], d["random_state"], missing_rate=d["missing_rate"]
                ).to_csv(raw_path, index=False)

            df = pd.read_csv(raw_path)
            logger.info("Dataset loaded: %s records", f"{len(df):,}")

            new_path = self.cfg.path("new_data")
            if include_new_data and new_path.exists():
                new_df = pd.read_csv(new_path)
                logger.info("Adding %s new records from %s", f"{len(new_df):,}", new_path.name)
                df = pd.concat([df, new_df], ignore_index=True)

            out = self.cfg.ensure_dir("processed_dir") / "ingested.csv"
            df.to_csv(out, index=False)
            logger.info("Data ingestion completed -> %s", out)
            return out
        except CustomException:
            raise
        except Exception as e:
            raise DataIngestionException(e, sys) from e
