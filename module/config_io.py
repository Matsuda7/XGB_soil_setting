"""Configuration reader without numerical-library imports."""
import json
import re
from pathlib import Path

def load_jsonc_config(path: Path) -> dict:
    """Read configuration with full-line comments and optional trailing commas."""
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        # JSONC-style full-line comments allow optional features to remain
        # visible without being selected. Inline comments are intentionally
        # unsupported so feature names containing // remain unambiguous.
        uncommented = "".join(
            line for line in stream if not line.lstrip().startswith("//")
        )
    # Also tolerate a comma left immediately before ] or } after a line is
    # commented out, which makes toggling individual feature lines convenient.
    uncommented = re.sub(r",(?=\s*[}\]])", "", uncommented)
    return json.loads(uncommented)

