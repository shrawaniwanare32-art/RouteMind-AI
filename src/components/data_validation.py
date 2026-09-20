"""Step 2 - schema, missing-value, range, duplicate and outlier checks."""
import sys
from pathlib import Path

import pandas as pd

from src.config.configuration import AppConfig
from src.utils.exception import CustomException, DataValidationException
from src.utils.helpers import save_json, utc_now_iso
from src.utils.logger import get_logger
from src.utils.schema import (
    CATEGORICAL_FEATURES,
    DELIVERY_ZONES,
    NUMERIC_INPUTS,
    RAW_FEATURES,
    TARGET_COL,
    VEHICLE_TYPES,
)

logger = get_logger("data_validation")


class DataValidation:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.vcfg = cfg["validation"]

    def run(self, ingested_path: Path) -> tuple[Path, dict]:
        try:
            logger.info("Data validation started")
            df = pd.read_csv(ingested_path)
            report: dict = {"created_at": utc_now_iso(), "rows_in": int(len(df))}

            # 1) schema
            required = RAW_FEATURES + [TARGET_COL]
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # 2) types / categories
            for col in NUMERIC_INPUTS + [TARGET_COL]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            valid = {"vehicle_type": VEHICLE_TYPES, "delivery_zone": DELIVERY_ZONES}
            for col in CATEGORICAL_FEATURES:
                s = df[col].astype(str).str.strip().str.lower()
                df[col] = s.where(s.isin(valid[col]))  # unknown category -> NaN (imputed later)

            # 3) duplicates
            if "order_id" in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset="order_id", keep="last")
            else:
                before = len(df)
                df = df.drop_duplicates()
            report["duplicates_removed"] = int(before - len(df))

            # 4) missing values
            ratios = df[required].isna().mean()
            report["missing_ratio"] = {c: round(float(r), 4) for c, r in ratios.items() if r > 0}
            too_missing = ratios[ratios > self.vcfg["max_missing_ratio"]]
            if len(too_missing) > 0:
                raise ValueError(f"Too many missing values in: {list(too_missing.index)}")

            # 5) target must exist and be sane
            bad_target = df[TARGET_COL].isna() | (df[TARGET_COL] < 0) | (
                df[TARGET_COL] > self.vcfg["max_delay_minutes"]
            )
            report["invalid_target_rows_removed"] = int(bad_target.sum())
            df = df[~bad_target]

            # 6) ranges - drop physically impossible rows (NaN is fine, imputed later)
            invalid = pd.Series(False, index=df.index)
            per_col = {}
            for col, (lo, hi) in self.vcfg["ranges"].items():
                bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
                per_col[col] = int(bad.sum())
                invalid |= bad
            report["out_of_range_rows_removed"] = int(invalid.sum())
            report["out_of_range_by_column"] = {c: n for c, n in per_col.items() if n}
            df = df[~invalid]

            # 7) outliers (report only, IQR rule)
            outliers = {}
            for col in ["distance_km", "current_speed", "driver_experience", TARGET_COL]:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                n_out = int(((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum())
                outliers[col] = n_out
            report["extreme_outliers_flagged"] = outliers

            if len(df) < self.vcfg["min_rows"]:
                raise ValueError(f"Only {len(df)} valid rows left (min {self.vcfg['min_rows']})")

            report["rows_out"] = int(len(df))
            out = self.cfg.ensure_dir("processed_dir") / "validated.csv"
            df.reset_index(drop=True).to_csv(out, index=False)
            save_json(report, self.cfg.ensure_dir("metrics_dir") / "validation_report.json")
            logger.info(
                "Data validation completed: %s -> %s rows", report["rows_in"], report["rows_out"]
            )
            return out, report
        except CustomException:
            raise
        except Exception as e:
            raise DataValidationException(e, sys) from e
