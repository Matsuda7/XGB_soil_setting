"""Project-relative paths, independent of the shell working directory."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TRAINING = ROOT / "data/training"
INTERIM = ROOT / "data/interim"
CACHE = ROOT / "data/cache"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"
def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
