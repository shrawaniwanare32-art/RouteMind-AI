"""Centralised logging -> console + logs/application.log (rotating)."""
import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config.configuration import load_config

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    logs_dir = load_config().ensure_dir("logs_dir")
    root = logging.getLogger("routemind")
    root.setLevel(logging.INFO)
    root.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(
        logs_dir / "application.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"routemind.{name}")
