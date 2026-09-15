"""Stage two: produce target-specific training tables from common raw data."""
import json
import pandas as pd
from module.paths import INTERIM, TRAINING, RESULTS, LOGS


def prepare_strength(targets):
    from module.soil_tests import load_test_records
    from module.paths import RAW
    from module.strength_distributions import MODEL_FEATURE_COLUMNS
    frame = load_test_records(RAW / 'soiltest/strength')
    (INTERIM / 'strength').mkdir(parents=True, exist_ok=True)
    frame.to_csv(INTERIM / 'strength/strength_data.csv', index=False, encoding='utf-8-sig')
    features = [c for c in MODEL_FEATURE_COLUMNS if c in frame]
    report = {}
    if 'gamma' in targets:
        from module.gamma_data import prepare
        report['gamma'] = prepare(raw=frame)
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
