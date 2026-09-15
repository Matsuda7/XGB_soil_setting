"""Read target configuration with shared spatial defaults; paths are project-relative."""
import json
from module.paths import project_path


def load_prediction_settings(path):
    config = json.loads(project_path(path).read_text(encoding='utf-8'))
    if 'spatial_config' in config:
        spatial = json.loads(project_path(config['spatial_config']).read_text(encoding='utf-8'))
        config = {**spatial, **config}
    return config
