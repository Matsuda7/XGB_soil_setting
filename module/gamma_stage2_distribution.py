"""Train a depth-enabled stage 2 model and predict a configured horizontal slice."""
import argparse
import json
import math

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from xgboost import XGBRegressor

from module.paths import project_path
from module.gamma_config import grid_shape
from module.gamma_features import load_dem, geographic_context, terrain_at, model_frame
from module.gamma_prediction import predict_grid
from module.gamma_stage2 import exclude_heldout, regenerate
from module.gamma_stage2_integrated import (
    SOLPHYJ_COLUMNS, build_missing_preprocessor, fit_evaluation_missing,
)
from module.solphyj_representative_soil import load_soil_map
from module.spatial_features import meshcodes_250m
from module.validation import split_data
from module.xgb_common import build_regressor, regression_metrics


def load_config(path):
    from module.configuration import load_prediction_settings
    config = load_prediction_settings(path)
    depth = config['prediction_depth_m']
    if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not math.isfinite(depth) or depth < 0:
        raise ValueError('prediction_depth_m must be finite and non-negative')
    if 'depth' not in config['numeric_features']:
        raise ValueError('Stage 2 requires depth as a predictor')
    base = {'x', 'y', 'elevation', 'depth', *SOLPHYJ_COLUMNS}
    variants = {
        'F3_depth': (base, set()),
        'F4_AVS_JCODE_depth': (base | {'slope', 'curvature', 'jshis_avs30'}, {'geology', 'jshis_jcode'}),
    }
    expected = variants.get(config['distribution_model'])
    if expected is None or (set(config['numeric_features']), set(config['categorical_features'])) != expected:
        raise ValueError('Features must match the selected depth-enabled model')
    if config['sample_weight'] != 1.0:
        raise ValueError('Expected equal weights')
    config['spatial'] = {key: config[key] for key in ('grid_input','dem_input','dem_cache','grid_crs','grid_spacing_m','property_mask','property_mask_spacing_m','property_mask_bounds')}
    grid_shape(config['spatial'])
    ratio = config['display_spacing_m'] / config['spatial']['grid_spacing_m']
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
        raise ValueError('Display spacing must be a positive multiple of grid spacing')
    return config


def attach_soil(frame, soil, representative, crs):
    points = gpd.GeoDataFrame({'_point': np.arange(len(frame))},
        geometry=gpd.points_from_xy(frame.x, frame.y), crs=crs)
    joined = gpd.sjoin(points, soil[['SSerGrCD', 'geometry']], how='left', predicate='within')
    ambiguous = joined.groupby('_point').SSerGrCD.nunique().gt(1)
    joined = joined.sort_values(['_point', 'index_right']).drop_duplicates('_point')
    joined.loc[joined._point.isin(ambiguous[ambiguous].index), 'SSerGrCD'] = pd.NA
    joined = joined.merge(representative[['SSG code', *SOLPHYJ_COLUMNS]],
        left_on='SSerGrCD', right_on='SSG code', how='left', validate='many_to_one').sort_values('_point')
    result = frame.reset_index(drop=True).copy()
    result[SOLPHYJ_COLUMNS] = joined[SOLPHYJ_COLUMNS].to_numpy()
    return result


class FeatureBuilder:
    """Use identical terrain, polygon and mesh joins for samples and grid points."""
    def __init__(self, config):
        self.config = config
        self.crs = config['spatial']['grid_crs']
        self.dem = load_dem(config)
        self.geology, self.jshis = geographic_context(config)
        self.geology = self.geology.to_crs(self.crs)
        self.soil = load_soil_map().to_crs(self.crs)
        self.soil['SSerGrCD'] = self.soil.SSerGrCD.astype('string')
        self.representative = pd.read_csv(project_path(config['representative_properties']), dtype={'SSG code': 'string'})
        self.transform = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

    def __call__(self, frame):
        result = frame.reset_index(drop=True).copy()
        terrain = terrain_at(result, self.dem, self.config)
        for col in ('slope', 'curvature'):
            result[col] = terrain[col].to_numpy()
        result['elevation'] = terrain.surface_z.to_numpy() - result.depth.to_numpy()
        if not np.isfinite(result[['x', 'y', 'depth', 'elevation', 'slope', 'curvature']]).all(axis=None):
            raise ValueError('Nonfinite coordinates/depth/terrain')
        points = gpd.GeoDataFrame({'_point': np.arange(len(result))},
            geometry=gpd.points_from_xy(result.x, result.y), crs=self.crs)
        joined = gpd.sjoin(points, self.geology[['symbol', 'geometry']], how='left', predicate='within')
        joined = joined.sort_values(['_point', 'index_right']).drop_duplicates('_point')
        result['geology'] = joined.symbol.astype('string').fillna('UNKNOWN').to_numpy()
        lon, lat = self.transform.transform(result.x.to_numpy(), result.y.to_numpy())
        codes = pd.DataFrame({'jshis_meshcode': meshcodes_250m(lon, lat)})
        attrs = codes.merge(self.jshis[['jshis_meshcode', 'jshis_avs30', 'jshis_jcode']],
                           how='left', on='jshis_meshcode', validate='many_to_one')
        invalid = attrs.jshis_jcode.isna() | attrs.jshis_jcode.eq(0) | attrs.jshis_avs30.fillna(0).le(0)
        attrs.loc[invalid, ['jshis_avs30', 'jshis_jcode']] = np.nan
        result[['jshis_avs30', 'jshis_jcode']] = attrs[['jshis_avs30', 'jshis_jcode']].to_numpy()
        return attach_soil(result, self.soil, self.representative, self.crs)


def train(config, builder):
    out = project_path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    frame = builder(pd.read_csv(project_path(config['input'])))
    if frame.record_id.duplicated().any() or not frame.label_source.isin(['measured', 'predicted']).all():
        raise ValueError('Invalid training identifiers/provenance')
    if not np.isfinite(frame.target).all() or (frame.target <= 0).any() or (frame.depth < 0).any():
        raise ValueError('Invalid gamma targets or depths')
    stage1 = json.loads(project_path(config['stage1_settings']).read_text())
    if not all('depth' in cols for cols in stage1['features'].values()):
        raise ValueError('Expected depth-enabled stage 1')
    measured = frame.loc[frame.label_source.eq('measured')].copy()
    candidates = frame.loc[frame.label_source.eq('predicted')].copy()
    frame.to_csv(out/'training_features.csv', index=False)
    numeric, categorical = config['numeric_features'], config['categorical_features']
    rows = []
    for mode in config['validation_modes']:
        for seed in config['split_seeds']:
            dest = out/f'{mode}_{seed}'; dest.mkdir()
            validation = dict(mode=mode, 
                test_size=config['test_size'], validation_size=config['validation_size'], random_state=seed)
            parts = split_data(measured, validation, target_column='target')
            learning, val, test = parts
            heldout = pd.concat([val, test])
            if set(zip(learning.x, learning.y)) & set(zip(heldout.x, heldout.y)):
                raise ValueError('Training/heldout coordinate overlap')
            safe = exclude_heldout(candidates, heldout)
            pseudo = regenerate(learning, safe, stage1['features'], stage1['config']['xgb_parameters'], config, seed, dest)
            augmented = pd.concat([learning, pseudo], ignore_index=True)
            overlap = len(set(augmented.boring_id) & set(heldout.boring_id))
            if overlap:
                raise ValueError('Training/heldout group overlap')
            audit = dict(test_labels_used_in_teacher=False, validation_labels_used_in_teacher=False,
                reused_saved_pseudo_targets=False, heldout_borehole_overlap=overlap)
            (dest/'leakage_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            pseudo.to_csv(dest/'regenerated_pseudo.csv', index=False)
            pd.concat([p[['record_id', 'boring_id']].assign(split=label)
                for p, label in zip(parts, ('train', 'validation', 'test'))]).to_csv(dest/'measured_split.csv', index=False)
            for condition, data in [('measured_only', learning), ('D_first_augmented', augmented)]:
                model, _, pred = fit_evaluation_missing((data, val, test), numeric, categorical, config['xgb_parameters'])
                rows.append(dict(mode=mode, seed=seed, condition=condition, train_count=len(data),
                    test_count=len(test), best_n_estimators=model.best_iteration+1,
                    **regression_metrics(test.target, pred)))
                test[['record_id', 'target', 'depth']].assign(prediction=pred).to_csv(dest/f'{condition}_predictions.csv', index=False)
            print(f'Completed {mode} seed={seed}', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'metrics.csv', index=False)
    summary = scores.groupby(['mode', 'condition'])[['r2', 'rmse', 'mae']].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    trees = int(scores.loc[scores['mode'].eq(config['primary_validation']) & scores.condition.eq('D_first_augmented'), 'best_n_estimators'].median())
    pre = build_missing_preprocessor(numeric, categorical)
    x = pre.fit_transform(model_frame(frame, numeric, categorical))
    params = dict(config['xgb_parameters'], n_estimators=trees)
    params.pop('early_stopping_rounds', None)
    model = build_regressor(params); model.fit(x, frame.target, verbose=False)
    final = out/'final'; final.mkdir()
    model.save_model(final/'model.json'); joblib.dump(pre, final/'preprocessor.joblib')
    metadata = dict(numeric_features=numeric, categorical_features=categorical,
        training_depth_range_m=[float(frame.depth.min()), float(frame.depth.max())],
        training_rows=len(frame), measured_rows=len(measured), pseudo_rows=len(candidates), trees=trees,
        target='gamma_wet', unit='kN/m3', evaluation='Heldout scores belong to evaluation models, not final fit')
    (final/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(summary.to_string(), flush=True)


def predict(config, builder, max_chunks=None):
    out = project_path(config['output'])
    saved = json.loads((out/'config.json').read_text())
    for key in ('numeric_features', 'categorical_features', 'spatial', 'representative_properties'):
        if config[key] != saved[key]:
            raise ValueError(f'Training/prediction configuration mismatch: {key}')
    metadata = json.loads((out/'final/metadata.json').read_text())
    model = XGBRegressor(); model.load_model(out/'final/model.json')
    pre = joblib.load(out/'final/preprocessor.joblib')
    def points_predictor(points, depth):
        frame = builder(points.assign(depth=float(depth)))
        values = model.predict(pre.transform(model_frame(frame, metadata['numeric_features'], metadata['categorical_features'])))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite gamma prediction')
        return frame[['x', 'y', 'elevation', 'depth']].assign(gamma_wet_kn_m3=values, density_wet_g_cm3=values/config['gravity_m_s2'])
    dest = out/f'grid_depth_{config["prediction_depth_m"]:g}m'
    report = predict_grid(config, {'wet': (model, pre, metadata)}, dest,
                          max_chunks=max_chunks, point_predictor=points_predictor)
    (dest/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/gamma/stage2_distribution.json')
    parser.add_argument('--stage', choices=['train', 'predict', 'all'], default='all')
    parser.add_argument('--max-chunks', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    builder = FeatureBuilder(config)
    if args.stage in ('train', 'all'):
        train(config, builder)
    if args.stage in ('predict', 'all'):
        report = predict(config, builder, args.max_chunks)
        if report['status'] != 'complete':
            raise SystemExit('Distribution incomplete; see prediction_summary.json')


if __name__ == '__main__':
    main()
