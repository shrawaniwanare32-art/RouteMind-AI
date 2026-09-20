"""Small shared helpers."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(obj, path: Path) -> None:
    """Atomic JSON write (temp file + rename) so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
