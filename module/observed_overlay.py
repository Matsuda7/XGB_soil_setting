"""Measured values near the prediction depth, independent of training filters."""
import numpy as np
import pandas as pd
from module.paths import RAW, project_path


def select_observations(frame, depth, tolerance, extent):
    """Inclusive depth window; nearest at each XY, averaging equal-distance ties."""
    frame = frame.copy()
    cols = ['x', 'y', 'depth', 'value']
    frame[cols] = frame[cols].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(frame[cols]).all(axis=1)
    frame = frame.loc[valid & frame.depth.ge(0) & frame.value.ge(0)
                      & frame.depth.between(depth-tolerance, depth+tolerance)
                      & frame.x.between(extent[0], extent[1])
                      & frame.y.between(extent[2], extent[3])].copy()
    frame['distance'] = (frame.depth-depth).abs()
    nearest = frame.groupby(['x', 'y']).distance.transform('min')
    frame = frame.loc[np.isclose(frame.distance, nearest, rtol=0, atol=1e-9)]
    return frame.groupby(['x', 'y'], as_index=False).agg(
        value=('value', 'mean'), depth=('depth', 'mean'),
        depth_min=('depth', 'min'), depth_max=('depth', 'max'), samples=('value', 'size'))


def raw_observations(target, config):
    if target == 'n':
        path = project_path(config['borehole_data'])
        wide = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        rows = []
        for number in range(1, 81):
            if f'深度{number}' in wide and f'N値{number}' in wide:
                rows.append(pd.DataFrame(dict(x=wide['坑口座標X'], y=wide['坑口座標Y'],
                    depth=wide[f'深度{number}'], value=wide[f'N値{number}'])))
        if not rows:
            raise ValueError('No measured N-value/depth columns found')
        return pd.concat(rows, ignore_index=True)
    from module.soil_tests import load_test_records, project_records
    source = RAW / 'soiltest/strength'
    raw = load_test_records(source)
    if target == 'c':
        from module.c_model_a import select_targets
        frame, _ = select_targets(raw)
        frame['value'] = frame.target
    else:
        from module.gamma_data import density_targets
        frame = density_targets(raw, config['gravity_m_s2'])
        frame['value'] = frame['gamma_' + target]
        frame['depth'] = pd.to_numeric(frame.depth_mid, errors='coerce')
    # No training depth, DEM, N-matching or train/test selection is applied.
    frame = frame.loc[frame.value.notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=['x', 'y', 'depth', 'value'])
    frame, _ = project_records(frame, source, config['spatial']['grid_crs'])
    return frame[['x', 'y', 'depth', 'value']]


def add_overlay(ax, image, target, config, csv_path):
    points = select_observations(raw_observations(target, config),
        config['prediction_depth_m'], config['observed_depth_tolerance_m'], image.get_extent())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(csv_path, index=False, encoding='utf-8-sig')
    depth, tolerance = config['prediction_depth_m'], config['observed_depth_tolerance_m']
    label = f'Observed: {depth:g} ± {tolerance:g} m ({len(points)} sites)'
    ax.scatter(points.x, points.y, c=points.value, cmap=image.cmap, norm=image.norm,
               s=35, marker='o', edgecolors='black', linewidths=.5, zorder=5, label=label)
    ax.legend(loc='best')
    # Keep the prediction footprint even when scatter autoscaling is enabled.
    extent = image.get_extent()
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    return points
