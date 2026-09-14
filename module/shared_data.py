"""Resolve optional shared raw-data paths without sharing generated outputs."""

import os
from pathlib import Path


def data_path(relative: str) -> Path:
    """Use XGB_DATA_ROOT when set, otherwise preserve project-local paths."""
    root = os.environ.get("XGB_DATA_ROOT", "").strip()
    return Path(root).expanduser() / relative if root else Path(relative)
