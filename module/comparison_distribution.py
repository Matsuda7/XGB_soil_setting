"""Refit after CV and stream identical prediction grids through both models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from module.paths import project_path
from module.comparison_data import load_target, TARGETS
from module.comparison_models import build_model, predict_batches, save_model
from module.tabpfn_runtime import batch_size
from module.xgb_common import build_preprocessor
from module.gamma_features import model_frame


def grid_chunks(config):
    options = config['distribution']
    if options.get('feature_csv'):
        yield from pd.read_csv(project_path(options['feature_csv']), chunksize=options['chunk_size'])
        return
    from module.gamma_features import load_dem, geographic_context, enrich
    from module.gamma_config import grid_shape
    from module.gamma_prediction import grid_indices
    from module.raster import read_binary_mask
    spatial = json.loads(project_path('config/spatial.json').read_text())
    cfg = dict(spatial=spatial, numeric_features=sorted({f for s in config['targets'].values() for f in s['numeric_features']}),
               categorical_features=sorted({f for s in config['targets'].values() for f in s['categorical_features']}))
    shape = grid_shape(spatial)
    mask = read_binary_mask(project_path(spatial['property_mask']), shape)
    seen = np.zeros(shape, dtype=bool)
    dem = load_dem(cfg)
    geology, jshis = geographic_context(cfg)
    reader = pd.read_csv(project_path(spatial['grid_input']), sep=r'\s+', header=None,
                         names=['x', 'y', 'legacy_z', 'legacy_bedrock'], usecols=[0, 1],
                         chunksize=options['chunk_size'])
    for points in reader:
        xi, yi, inside = grid_indices(points, spatial, shape)
        active = np.zeros(len(points), dtype=bool)
        active[inside] = mask[yi[inside], xi[inside]] != 0
        points = points.loc[active].reset_index(drop=True)
        xi, yi = xi[active], yi[active]
        if points.empty:
            continue
        if len(np.unique(yi * shape[1] + xi)) != len(points) or seen[yi, xi].any():
            raise ValueError('Duplicate prediction coordinates')
        seen[yi, xi] = True
        points['depth'] = options['depth_m']
        yield enrich(points, cfg, dem, geology, jshis)
    if ((mask != 0) & ~seen).any():
        raise ValueError('Input grid misses active mask cells')


def distribute(config, output):
    from module.model_comparison import dense, write_json, sha256
    options = config['distribution']
    spatial = json.loads(project_path('config/spatial.json').read_text())
    if options['depth_m'] < 0 or options['chunk_size'] < 1:
        raise ValueError('Invalid distribution depth or chunk size')
    if options.get('feature_csv'):
        source = project_path(options['feature_csv'])
        source_info = dict(feature_csv=str(source), sha256=sha256(source))
    else:
        source_info = {k: sha256(project_path(spatial[k])) for k in ('grid_input', 'dem_input', 'property_mask')}
    write_json(output / 'distribution_config.json', dict(options=options, spatial=spatial, inputs=source_info,
               correction='raw predictions preserved; only N-to-phi uses N clipped to [0,50] and existing phi cap',
               differences='TabPFN minus XGBoost on raw predictions'))
    fitted = {}
    for name, spec in config['targets'].items():
        metadata_path = output / name / 'dataset.json'
        if metadata_path.exists():
            expected = json.loads(metadata_path.read_text())['source_sha256']
            if sha256(project_path(spec['data'])) != expected:
                raise ValueError(f'{name}: input data changed since CV')
        data, _, target, unit = load_target(name, spec, config)
        numeric, categorical = spec['numeric_features'], spec['categorical_features']
        pre = build_preprocessor(numeric, categorical)
        values = dense(pre.fit_transform(model_frame(data, numeric, categorical)))
        for model_name in config['models']:
            dest = output / name / model_name / 'final'
            dest.mkdir(parents=True)
            model = build_model(model_name, config, len(values), values.shape[1])
            model.fit(values, data[target].to_numpy())
            save_model(model, model_name, dest)
            joblib.dump(pre, dest / 'preprocessor.joblib')
            write_json(dest / 'metadata.json', dict(rows=len(data), unit=unit, parameters=model.get_params(),
                       training_depth_range=[data.depth.min(), data.depth.max()]))
            fitted[name, model_name] = (model, pre)
    total = 0
    invalid_counts = {}
    display = []
    seen = set() if options.get('feature_csv') else None
    for number, frame in enumerate(grid_chunks(config), 1):
        if not np.isfinite(frame[['x', 'y', 'depth']].to_numpy()).all():
            raise ValueError('Non-finite grid coordinates/depth')
        if not np.allclose(frame.depth, options['depth_m'], rtol=0, atol=1e-8):
            raise ValueError('Grid depth differs from distribution depth')
        spacing = spatial['grid_spacing_m']
        origin = [spatial['property_mask_bounds']['xmin'], spatial['property_mask_bounds']['ymin']]
        indices = (frame[['x', 'y']].to_numpy() - origin) / spacing
        if not np.allclose(indices, np.round(indices), rtol=0, atol=1e-6):
            raise ValueError('Coordinates do not match configured mesh resolution/origin')
        if seen is not None:
            keys = list(map(tuple, frame[['x', 'y', 'depth']].to_numpy()))
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError('Duplicate grid coordinates')
            seen.update(keys)
        for name, spec in config['targets'].items():
            predictions = {}
            for model_name in config['models']:
                model, pre = fitted[name, model_name]
                features = model_frame(frame, spec['numeric_features'], spec['categorical_features'])
                values = predict_batches(model, dense(pre.transform(features)), batch_size(model_name, config),
                                         output / name / model_name / 'prediction_runtime.jsonl'
                                         if model_name == 'tabpfn' else None)
                predictions[model_name] = values
                table = frame[['x', 'y', 'depth']].copy()
                table['predicted_raw'] = values
                invalid = (values < 0) | ((values > 50) if name == 'N' else False)
                key = f'{name}/{model_name}'
                invalid_counts[key] = invalid_counts.get(key, 0) + int(invalid.sum())
                if name == 'N':
                    from module.phi_prediction import PHI_MAX_DEG
                    table['predicted_phi_deg'] = np.minimum(np.sqrt(15 * np.clip(values, 0, 50)) + 15, PHI_MAX_DEG)
                dest = output / name / model_name / 'grid'
                dest.mkdir(exist_ok=True)
                table.to_csv(dest / f'{number:06d}.csv.gz', index=False)
                # Display every 100 m on the common mesh; prediction CSVs retain every cell.
                display_mask = (np.round(indices).astype(int) % 10 == 0).all(axis=1)
                display.append(table.loc[display_mask, ['x', 'y', 'predicted_raw']].assign(target=name, model=model_name))
            if set(predictions) == {'xgboost', 'tabpfn'}:
                table = frame[['x', 'y', 'depth']].copy()
                table['difference_raw'] = predictions['tabpfn'] - predictions['xgboost']
                dest = output / name / 'difference'
                dest.mkdir(exist_ok=True)
                table.to_csv(dest / f'{number:06d}.csv.gz', index=False)
        total += len(frame)
    if total == 0:
        raise ValueError('No prediction grid rows')
    if display:
        from module.comparison_reporting import distribution_plots
        distribution_plots(pd.concat(display, ignore_index=True), output)
    write_json(output / 'distribution_summary.json', dict(status='complete', rows=total,
               depth_m=options['depth_m'], format='csv.gz', invalid_raw_counts=invalid_counts,
               units={n: TARGETS[n][1] for n in config['targets']}))
