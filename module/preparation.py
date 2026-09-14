"""Stage two: produce target-specific training tables from common raw data."""
import json
import pandas as pd
from module.paths import INTERIM, TRAINING, RESULTS, LOGS


def prepare_strength(targets):
    from module.strength import run, DEFAULT_INPUT_DIR
    from module.strength_distributions import MODEL_FEATURE_COLUMNS, GRAVITY
    summary = run(DEFAULT_INPUT_DIR, INTERIM / 'strength', LOGS / 'strength')
    if summary['xml_files_error']:
        raise ValueError(f"{summary['xml_files_error']} XML files failed; see logs/strength/error.log")
    frame = pd.read_csv(INTERIM / 'strength/strength_data.csv', encoding='utf-8-sig')
    features = [c for c in MODEL_FEATURE_COLUMNS if c in frame]
    report = {}
    if 'gamma' in targets:
        counts = {}
        for label in ('wet', 'dry'):
            density = pd.to_numeric(frame[f'{label}_density'], errors='coerce')
            valid = density.notna()
            dataset = frame.loc[valid, features].copy()
            dataset.insert(0, 'target', density.loc[valid] * GRAVITY)
            path = TRAINING / f'gamma/{label}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(path, index=False, encoding='utf-8-sig')
            counts[label] = len(dataset)
        report['gamma'] = {'status': 'prepared' if any(counts.values()) else 'no_data', 'rows': counts,
                           'target_unit': 'kN/m3', 'assumed_density_unit': 'g/cm3'}
    if 'c' in targets:
        # Preserve the distinction between cohesion and measured shear strength.
        cohesion = pd.to_numeric(frame['c_total'], errors='coerce')
        valid = cohesion.notna()
        dataset = frame.loc[valid, features + ['unit_c']].copy()
        dataset.insert(0, 'target', cohesion.loc[valid])
        path = TRAINING / 'c/candidates.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False, encoding='utf-8-sig')
        report['c'] = {'status': 'needs_definition' if len(dataset) else 'no_data', 'rows': len(dataset),
                       'reason': 'Cohesion definition, units and training policy must be confirmed; shear strength is not substituted.'}
    return report


def prepare_phi():
    from module import boring, spatial, spt_dataset
    boring.main()
    spatial.main()
    spt_dataset.main()
    return {'status': 'prepared', 'dataset': 'data/training/phi/model_dataset.csv'}
