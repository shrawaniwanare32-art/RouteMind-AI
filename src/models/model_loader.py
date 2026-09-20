"""Model registry: versioned artifacts on disk + a `latest.json` pointer.

artifacts/models/
    v20260920123000123/
        classifier.joblib
        regressor.joblib
        metadata.json
    latest.json        <- {"version": "v2026..."}  = the production model
"""
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config.configuration import AppConfig
from src.utils.helpers import load_json, save_json, utc_now_iso


@dataclass
class ModelBundle:
    version: str
    classifier: object
    regressor: object
    metadata: dict


class ModelRegistry:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.models_dir = cfg.ensure_dir("models_dir")
        self.pointer = self.models_dir / "latest.json"

    def _new_version_id(self) -> str:
        base = "v" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
        version, i = base, 1
        while (self.models_dir / version).exists():
            version, i = f"{base}-{i}", i + 1
        return version

    def save_candidate(
        self, clf, reg, metrics: dict, extra: dict | None = None, reference: pd.DataFrame | None = None
    ) -> str:
        version = self._new_version_id()
        vdir = self.models_dir / version
        vdir.mkdir(parents=True)
        joblib.dump(clf, vdir / "classifier.joblib")
        joblib.dump(reg, vdir / "regressor.joblib")
        if reference is not None:
            reference.to_csv(vdir / "reference.csv", index=False)
        save_json(
            {"version": version, "created_at": utc_now_iso(), "metrics": metrics, **(extra or {})},
            vdir / "metadata.json",
        )
        return version

    def promote(self, version: str) -> None:
        if not (self.models_dir / version).exists():
            raise FileNotFoundError(f"Unknown model version: {version}")
        ref = self.models_dir / version / "reference.csv"
        if ref.exists():  # the drift baseline always describes the PRODUCTION model's training data
            ref_dir = self.cfg.ensure_dir("reference_dir")
            shutil.copyfile(ref, ref_dir / "reference.csv")
        save_json({"version": version, "promoted_at": utc_now_iso()}, self.pointer)

    def latest_version(self) -> str | None:
        if not self.pointer.exists():
            return None
        return load_json(self.pointer).get("version")

    def load(self, version: str | None = None) -> ModelBundle:
        version = version or self.latest_version()
        if version is None:
            raise FileNotFoundError("No production model found. Run the training pipeline first.")
        vdir = self.models_dir / version
        return ModelBundle(
            version=version,
            classifier=joblib.load(vdir / "classifier.joblib"),
            regressor=joblib.load(vdir / "regressor.joblib"),
            metadata=load_json(vdir / "metadata.json"),
        )

    def list_versions(self) -> list[dict]:
        out = []
        current = self.latest_version()
        for p in sorted(self.models_dir.glob("v*/metadata.json")):
            meta = load_json(p)
            meta["is_production"] = meta["version"] == current
            out.append(meta)
        return out

    def delete_version(self, version: str) -> None:
        if version == self.latest_version():
            raise ValueError("Refusing to delete the production model")
        shutil.rmtree(self.models_dir / version, ignore_errors=True)
