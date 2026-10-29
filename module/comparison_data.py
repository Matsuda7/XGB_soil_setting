"""Measured target inputs and common, auditable spatial folds."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from module.paths import project_path

# Only exogenous map/terrain/position predictors in the first comparison protocol.
# Learned N, laboratory properties, target aliases and pseudo-labels are excluded.
NUMERIC = {'x', 'y', 'depth', 'sample_elevation', 'surface_z', 'slope', 'curvature',
           'jshis_avs30', 'jshis_arv'}
CATEGORICAL = {'symbol', 'ser', 'jshis_jcode', 'soil_inventory_ssg'}
TARGETS = {'c': ('target', 'kPa'), 'N': ('n_value', 'dimensionless'),
           'gamma_wet': ('target', 'kN/m3'), 'gamma_dry': ('target', 'kN/m3')}


def load_target(name, spec, config):
    target, unit = TARGETS[name]
    numeric, categorical = spec['numeric_features'], spec['categorical_features']
    if not numeric or len(set(numeric + categorical)) != len(numeric + categorical):
        raise ValueError('Features must be nonempty and unique')
    if set(numeric) - NUMERIC or set(categorical) - CATEGORICAL:
        raise ValueError('Only independently observed geographic features are supported; '
                         'predicted N / pseudo-data require a nested spatial feature generator')
    frame = pd.read_csv(project_path(spec['data']), dtype={'boring_id': str})
    frame['source_row_id'] = np.arange(len(frame))
    required = {target, 'boring_id', 'x', 'y', 'depth', *numeric, *categorical}
    # N's existing loader attaches independent map attributes and applies depth/N filters.
    if name == 'N' and (set(categorical) - set(frame)):
        from module.phi_training import load_dataset
        frame = load_dataset(project_path(spec['data']), numeric, categorical,
                             training_min_depth_m=None, training_max_depth_m=None)
        frame['source_row_id'] = np.arange(len(frame))
    if required - set(frame):
        raise ValueError(f'{name}: missing columns {sorted(required - set(frame))}')
    minimum, maximum = config['training_depth_m']
    keep = frame.depth.between(minimum, maximum)
    if name == 'N' and not config.get('include_n50', True):
        keep &= frame[target].ne(50)
    excluded = frame.loc[~keep, ['source_row_id', 'boring_id', 'depth']].copy()
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty or frame[['boring_id', target, 'x', 'y', 'depth']].isna().any(axis=None):
        raise ValueError(f'{name}: empty data or missing ID/target/coordinates/depth')
    if not np.isfinite(frame[[target, 'x', 'y', 'depth']].to_numpy()).all():
        raise ValueError('Non-finite targets or coordinates')
    if name == 'N' and not frame[target].between(0, 50).all():
        raise ValueError('N targets must follow existing [0,50] preparation')
    if name == 'c':
        sources = {'explicit_c_effective', 'soiltestlist_effective_strength_intercept'}
        if 'target_source' not in frame or not frame.target_source.isin(sources).all():
            raise ValueError('c comparison requires measured effective cohesion (c_effective)')
        if not frame[target].ge(0).all() or 'target_unit' not in frame or not frame.target_unit.eq('kPa').all():
            raise ValueError('c requires nonnegative targets in kPa')
    if name.startswith('gamma') and not frame[target].gt(0).all():
        raise ValueError('Unit weight must be positive')
    frame['row_id'] = np.arange(len(frame))
    return frame, excluded, target, unit


def spatial_folds(frame, config):
    """10 km blocks, unioned when any borehole ID spans multiple blocks.

    This preserves both spatial blocks and complete holes, including source aliases.
    Block membership is independent of labels. Origin is fixed in EPSG:6675.
    """
    if config['crs'] != 'EPSG:6675' or config['block_size_m'] != 10000:
        raise ValueError('This protocol requires EPSG:6675 and 10000 m blocks')
    xy = frame[['x', 'y']].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or frame.boring_id.isna().any():
        raise ValueError('Finite coordinates and complete borehole IDs required')
    block = pd.Series(['%d:%d' % tuple(v) for v in np.floor(xy / 10000).astype(int)], index=frame.index)
    parent = {b: b for b in block.unique()}
    def root(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b
    for column in ('boring_id', 'source_boring_id', 'comparison_hole_id'):
        if column in frame:
            for _, indices in frame.groupby(column).groups.items():
                members = block.loc[indices].unique()
                for b in members[1:]:
                    a, c = root(members[0]), root(b)
                    parent[max(a, c)] = min(a, c)
    groups = block.map(root)
    unique = np.array(sorted(groups.unique()))
    folds = config['n_splits']
    if type(folds) is not int or folds < 2 or len(unique) < folds:
        raise ValueError(f'{len(unique)} independent spatial groups cannot support {folds} folds')
    assignments = np.zeros(len(frame), dtype=int)
    for fold, (_, test_groups) in enumerate(KFold(folds, shuffle=True, random_state=config['random_state']).split(unique), 1):
        assignments[groups.isin(unique[test_groups])] = fold
    result = frame[['row_id', 'source_row_id', 'boring_id', 'x', 'y', 'depth']].copy()
    result['block'] = block.to_numpy()
    result['spatial_group'] = groups.to_numpy()
    result['fold'] = assignments
    for fold in range(1, folds + 1):
        test = assignments == fold
        if test.sum() < 2 or (~test).sum() < 2:
            raise ValueError(f'Fold {fold} requires at least two train and test rows')
        if set(frame.loc[test, 'boring_id']) & set(frame.loc[~test, 'boring_id']):
            raise RuntimeError('Borehole leakage')
    return result
