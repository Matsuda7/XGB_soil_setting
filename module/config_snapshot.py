"""Write resolved run settings consistently for c, gamma and phi."""
from copy import deepcopy
import json
from pathlib import Path
from module.paths import project_path

SPATIAL_KEYS = ('grid_input', 'dem_input', 'dem_cache', 'grid_crs',
                'grid_spacing_m', 'property_mask', 'property_mask_spacing_m',
                'property_mask_bounds')


def write_config_snapshot(config, output, target):
    """Snapshot settings without mutating the configuration used for computation."""
    snapshot = deepcopy(config)
    snapshot['output_dir'] = str(output)
    if target == 'phi':
        from module import phi_training
        from module.configuration import load_model_settings
        from module.xgb_common import load_model_config
        numeric, categorical = load_model_config(phi_training.MODEL_CONFIG)
        snapshot.update(
            model_config=str(phi_training.MODEL_CONFIG),
            model_settings=load_model_settings(),
            numeric_features=numeric, categorical_features=categorical,
            validation=phi_training.load_validation_config(),
            xgb_parameters=dict(phi_training.XGB_PARAMETERS),
            training_dataset=str(phi_training.INPUT_CSV),
            spatial={key: config[key] for key in SPATIAL_KEYS},
        )
    if target == 'gamma' and snapshot.get('distribution_model') == 'predicted_n':
        snapshot['numeric_features'] = list(dict.fromkeys([*snapshot['numeric_features'], 'n_value']))
    if 'spatial' in snapshot:
        for key in ('grid_input', 'dem_input', 'dem_cache', 'property_mask'):
            snapshot['spatial'][key] = str(project_path(snapshot['spatial'][key]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return snapshot
