"""Load shared and target-specific model/prediction settings."""
import json
import math
from module.paths import project_path


def _read_json(path):
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def load_model_settings(path="config/model.json"):
    """Read the common inclusive training-depth range and N=50 policy."""
    settings = _read_json(path)
    minimum = settings.get("training_min_depth_m")
    maximum = settings.get("training_max_depth_m")
    for name, value in (("training_min_depth_m", minimum), ("training_max_depth_m", maximum)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be null or a finite non-negative number")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("training_min_depth_m must not exceed training_max_depth_m")
    if not isinstance(settings.get("include_n50", True), bool):
        raise ValueError("include_n50 must be true or false")
    settings.setdefault("include_n50", True)
    return settings


def load_prediction_settings(path):
    """Merge spatial, shared prediction, then target-specific settings."""
    shared = _read_json("config/prediction.json")
    target = _read_json(path)
    config = {**shared, **target}
    spatial_path = config.get("spatial_config")
    if spatial_path:
        config = {**_read_json(spatial_path), **config}
    model = _read_json("config/model.json")
    for key in ("prediction_depth_m", "observed_depth_tolerance_m"):
        value = model.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative in config/model.json")
        config[key] = value
    return config
