"""Configurable holdout evaluation. No analysis runs during import."""
import json
import math
import numpy as np
from module.paths import project_path
from module.xgb_common import borehole_train_validation_test_split


def validate_config(config):
    if config.get('mode') not in ('borehole', 'spatial', 'row'):
        raise ValueError('validation.mode must be borehole, spatial or row')
    if not isinstance(config.get('additional_spatial'), bool):
        raise ValueError('additional_spatial must be boolean')
    size = config.get('block_size_m')
    if isinstance(size, bool) or not isinstance(size, (float, int)) or not math.isfinite(size) or size <= 0:
        raise ValueError('block_size_m must be positive and finite')
    for key in ('test_size', 'validation_size'):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value < 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if isinstance(config.get('random_state'), bool) or not isinstance(config.get('random_state'), int):
        raise ValueError('random_state must be an integer')
    return config


def load_config(path='config/phi/validation.json'):
    return validate_config(json.loads(project_path(path).read_text()))


def split_data(data, config, target_column='n_value'):
    validate_config(config)
    frame = data.copy()
    mode = config['mode']
    if frame.boring_id.isna().any():
        raise ValueError('Missing borehole IDs')
    if mode == 'spatial':
        if not np.isfinite(frame[['x', 'y']].to_numpy(dtype=float)).all():
            raise ValueError('Spatial validation requires finite projected coordinates in metres')
        # Multiple coordinates recorded for one hole use its median location.
        xy = frame.groupby('boring_id')[['x', 'y']].transform('median')
        blocks = np.floor(xy / config['block_size_m']).astype('int64')
        frame['_validation_group'] = blocks.x.astype(str) + ':' + blocks.y.astype(str)
    elif mode == 'row':
        frame['_validation_group'] = np.arange(len(frame)).astype(str)
    else:
        frame['_validation_group'] = frame.boring_id.astype(str)
    options = {k: config[k] for k in ('test_size', 'validation_size', 'random_state')}
    parts = borehole_train_validation_test_split(frame, target_column=target_column,
            id_column='_validation_group', **options)
    if mode != 'row':
        ids = [set(p.boring_id) for p in parts]
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            raise RuntimeError('Borehole leakage across validation splits')
    return parts


def save_split(parts, path):
    import pandas as pd
    rows = [p[['boring_id', 'x', 'y', '_validation_group']].drop_duplicates().assign(split=name)
            for name, p in zip(('train', 'validation', 'test'), parts)]
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
