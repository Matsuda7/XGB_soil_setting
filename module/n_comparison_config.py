"""Validated N-comparison options. No data or models are evaluated here."""
import json
import math
from module.paths import project_path
from module.xgb_common import load_model_config

SCENARIOS={'no_n','measured_n','measured_to_predicted_n','predicted_n',
           'predicted_n_reliability','measured_plus_supplemental','no_n_with_supplemental'}
ALLOWED_NUMERIC={'x','y','depth','surface_z','sample_z','n_value_elevation',
                 'slope','curvature','dem_missing','jshis_avs30','jshis_arv'}
ALLOWED_CATEGORICAL={'symbol','ser','jshis_jcode'}


def load_options(path):
    options=json.loads(project_path(path).read_text())
    if not isinstance(options.get('enabled'),bool): raise ValueError('N comparison enabled must be boolean')
    if not options['enabled']: return options
    numeric,categorical=load_model_config(project_path(options['model_config']))
    if set(numeric)-ALLOWED_NUMERIC or set(categorical)-ALLOWED_CATEGORICAL:
        raise ValueError('N comparison supports target-independent geographic features only; RBF/derived borehole targets are not supported')
    options['numeric_features']=numeric;options['categorical_features']=categorical
    for key in ('xy_tolerance_m','depth_tolerance_m','nearby_radius_m','supplemental_weight'):
        value=options.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'N comparison {key} must be finite and positive')
    value=options.get('interval_coverage')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<value<1:
        raise ValueError('interval_coverage must be between zero and one')
    folds=options.get('cross_fit_folds')
    if isinstance(folds,bool) or not isinstance(folds,int) or folds<2: raise ValueError('cross_fit_folds must be >=2')
    seeds=options.get('ensemble_seeds')
    if not isinstance(seeds,list) or len(seeds)<2 or any(isinstance(s,bool) or not isinstance(s,int) for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Use at least two distinct integer ensemble_seeds')
    scenarios=options.get('scenarios')
    if not isinstance(scenarios,list) or not scenarios or any(not isinstance(s,str) or s not in SCENARIOS for s in scenarios) or len(set(scenarios))!=len(scenarios):
        raise ValueError('Invalid or duplicate N comparison scenarios')
    from module.phi_training import XGB_PARAMETERS
    options['xgb_parameters']=dict(XGB_PARAMETERS)
    return options
