"""
Utility helpers for SDN Controller Placement framework.

Provides shared functionality such as logging setup and directory management.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with a consistent format.

    Parameters
    ----------
    level : str | None
        Logging level override (e.g. ``"DEBUG"``).  Falls back to the
        value defined in ``config.LOG_LEVEL``.
    """
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def ensure_directory(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not exist, then return it.

    Parameters
    ----------
    path : Path | str
        Directory path to ensure.

    Returns
    -------
    Path
        The resolved directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
