"""Compare laboratory predictors on measured wet gamma, with borehole holdouts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from module.gamma_data import density_targets
from module.gamma_features import load_dem, terrain_at
from module.gamma_training import fit_evaluation
from module.soil_tests import load_test_records, project_records
from module.validation import split_data
from module.xgb_common import regression_metrics

GROUPS = {
    'grain_size': ['gravel_fraction', 'sand_fraction', 'silt_fraction', 'clay_fraction'],
    'water_content': ['water_content'],
    'saturation': ['degree_of_saturation'],
    'void_ratio': ['void_ratio'],
}


def main():
    out = Path(__file__).resolve().parent
    config = json.loads((ROOT / 'config/gamma/prediction.json').read_text())
    config['spatial'] = json.loads((ROOT / config['spatial_config']).read_text())
    raw = density_targets(load_test_records(ROOT / 'data/raw/soiltest/strength'))
    raw['record_id'] = np.arange(len(raw))
    raw['depth'] = pd.to_numeric(raw.depth_mid, errors='coerce')
    raw = raw.drop_duplicates(['xml_file', 'sample_id', 'sample_no', 'depth', 'wet_density', 'dry_density'])
    raw = raw.loc[np.isfinite(raw.depth) & (raw.depth >= 0)].copy()
    frame, _ = project_records(raw, ROOT / 'data/raw/soiltest/strength', config['spatial']['grid_crs'])
    terrain = terrain_at(frame, load_dem(config), config)
    frame['elevation'] = terrain.surface_z.to_numpy() - frame.depth.to_numpy()
    frame['target'] = frame.gamma_wet
    frame = frame.loc[np.isfinite(frame[['x', 'y', 'elevation']]).all(axis=1)].copy()
    frame.to_csv(out / 'eligible_records.csv', index=False)
    rows = []
    for name, laboratory in GROUPS.items():
        features = ['x', 'y', 'elevation', *laboratory]
        values = frame[features].apply(pd.to_numeric, errors='coerce')
        available = np.isfinite(values).all(axis=1)
        subset = frame.loc[available].copy()
        subset[features] = values.loc[available]
        measured = subset.loc[subset.target.notna()].copy()
        candidates = subset.loc[subset.target.isna() & subset.wet_density.isna()].copy()
        measured.to_csv(out / f'{name}_measured.csv', index=False)
        candidates.to_csv(out / f'{name}_candidates.csv', index=False)
        for mode in ('borehole', 'spatial'):
            settings = dict(config['validation'], mode=mode)
            parts = split_data(measured, settings, target_column='target')
            model, _, prediction = fit_evaluation(parts, features, [], config['xgb_parameters'])
            # Position-only comparator uses exactly the same held-out samples.
            _, _, baseline = fit_evaluation(parts, ['x', 'y', 'elevation'], [], config['xgb_parameters'])
            row = dict(dataset=name, validation=mode, measured_wet_count=len(measured),
                       candidate_count=len(candidates), potential_total_count=len(measured)+len(candidates),
                       borehole_count=measured.boring_id.nunique(),
                       train_count=len(parts[0]), validation_count=len(parts[1]), test_count=len(parts[2]),
                       position_only_r2=regression_metrics(parts[2].target, baseline)['r2'],
                       best_n_estimators=model.best_iteration+1,
                       **regression_metrics(parts[2].target, prediction))
            rows.append(row)
            test = parts[2][['record_id', 'boring_id', *features, 'target']].copy()
            test['prediction'] = prediction
            test['position_only_prediction'] = baseline
            test.to_csv(out / f'{name}_{mode}_predictions.csv', index=False)
            pd.concat([part[['record_id', 'boring_id']].assign(split=split)
                       for split, part in zip(('train', 'validation', 'test'), parts)]).to_csv(
                           out / f'{name}_{mode}_split.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'comparison.csv', index=False)
    (out / 'comparison_settings.json').write_text(json.dumps(
        dict(config=config, feature_groups=GROUPS, missing_features='complete cases only',
             elevation='DEM surface_z minus sample depth',
             candidate_definition='wet density unmeasured; all predictors finite',
             evaluation='measured targets only; separate holdouts per dataset'), indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
