"""Gamma-specific model/depth configuration; shared geographic inputs with phi."""
import json
import math
from module.paths import CONFIG, project_path
from module.xgb_common import load_model_config

KINDS = ('wet', 'dry')
NUMERIC = {'x','y','depth','surface_z','sample_z','slope','curvature','jshis_avs30','jshis_arv','dem_missing'}
CATEGORICAL = {'symbol','jshis_jcode','ser'}


def grid_shape(spatial):
    bounds = spatial['property_mask_bounds']
    spacing = float(spatial['grid_spacing_m'])
    if not math.isfinite(spacing) or spacing <= 0 or spacing != float(spatial['property_mask_spacing_m']):
        raise ValueError('Gamma grid and prop spacing must be the same positive value')
    sizes = [(float(bounds[hi])-float(bounds[lo]))/spacing for lo,hi in [('ymin','ymax'),('xmin','xmax')]]
    if any(not math.isfinite(n) or n < 1 or not math.isclose(n, round(n), abs_tol=1e-7) for n in sizes):
        raise ValueError('Invalid gamma grid bounds')
    return tuple(int(round(n))+1 for n in sizes)


def load_config(path=None):
    config = json.loads(project_path(path or CONFIG/'gamma/prediction.json').read_text(encoding='utf-8'))
    from module.n_comparison_config import load_options
    config['n_comparison'] = load_options(config['n_comparison_config'])
    selected = config.get('distribution_model', 'no_n')
    if selected not in ('no_n', 'predicted_n'):
        raise ValueError('distribution_model must be no_n or predicted_n')
    config['distribution_model'] = selected
    if selected == 'predicted_n' and (not config['n_comparison']['enabled'] or 'predicted_n' not in config['n_comparison']['scenarios']):
        raise ValueError('predicted_n distribution requires enabled N comparison including predicted_n')
    if not isinstance(config.get('create_distribution'), bool):
        raise ValueError('create_distribution must be boolean')
    if config['n_comparison']['enabled'] and config['validation']['mode'] == 'row':
        raise ValueError('N comparison requires borehole/spatial validation; disable N comparison to use row validation')
    from module.validation import validate_config
    validate_config(config['validation'])
    depth = config['prediction_depth_m']
    if isinstance(depth, bool) or not isinstance(depth, (float,int)) or not math.isfinite(depth) or depth < 0:
        raise ValueError('prediction_depth_m must be a finite non-negative number')
    if config['density_unit'] != 'g/cm3' or float(config['gravity_m_s2']) != 9.80665:
        raise ValueError('Supported conversion: density in g/cm3 multiplied by 9.80665 -> kN/m3')
    if not isinstance(config['exclude_outside_dem'],bool) or not isinstance(config['write_text_matrix'],bool):
        raise ValueError('exclude_outside_dem and write_text_matrix must be booleans')
    size = config['input_chunk_size']
    if isinstance(size,bool) or not isinstance(size,int) or size < 1:
        raise ValueError('input_chunk_size must be a positive integer')
    spatial = json.loads(project_path(config['spatial_config']).read_text(encoding='utf-8'))
    grid_shape(spatial)
    if config['n_comparison']['enabled'] and spatial['grid_crs'] != 'EPSG:6675':
        raise ValueError('The SPT coordinate CSV uses EPSG:6675; transform it before using another CRS')
    display = float(config['display_spacing_m'])
    ratio = display / float(spatial['grid_spacing_m'])
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
        raise ValueError('display_spacing_m must be a positive multiple of grid spacing')
    numeric, categorical = load_model_config(project_path(config['model_config']))
    if set(numeric)-NUMERIC or set(categorical)-CATEGORICAL:
        raise ValueError('Gamma features must be available at prediction grid points; density and laboratory properties are not allowed')
    config['numeric_features'], config['categorical_features'] = numeric, categorical
    config['spatial'] = spatial
    return config


def data_signature(config):
    spatial = config['spatial']
    return {k:config[k] for k in ('density_unit','gravity_m_s2','exclude_outside_dem')} | {
        'n_comparison':config.get('n_comparison', {'enabled':False}),
        'spatial':{k:spatial[k] for k in ('grid_crs','dem_input','dem_cache','grid_spacing_m','property_mask_spacing_m','property_mask_bounds')}}
