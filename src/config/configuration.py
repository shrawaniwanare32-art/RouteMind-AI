"""Configuration loader.

* config/config.yaml holds all settings.
* $ROUTEMIND_HOME (optional) moves data/, artifacts/ and logs/ somewhere else
  (used by the tests and by containers with mounted volumes).
"""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_root() -> Path:
    return Path(os.getenv("ROUTEMIND_HOME", str(PROJECT_ROOT)))


class AppConfig:
    def __init__(self, raw: dict, root: Path):
        self.raw = raw
        self.root = root

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Absolute path for an entry of the `paths:` section."""
        return self.root / self.raw["paths"][key]

    def ensure_dir(self, key: str) -> Path:
        p = self.path(key)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config() -> AppConfig:
    cfg_path = Path(os.getenv("ROUTEMIND_CONFIG", str(PROJECT_ROOT / "config" / "config.yaml")))
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if os.getenv("ROUTEMIND_N_SAMPLES"):
        raw["data"]["n_samples"] = int(os.environ["ROUTEMIND_N_SAMPLES"])
    return AppConfig(raw, get_root())
