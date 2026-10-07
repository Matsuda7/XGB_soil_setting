"""Compact public plots and a single evaluation table for phi runs."""
import json
from pathlib import Path
import shutil

import pandas as pd

PLOTS = {
    'feature_contribution.png',
    'feature_importance.png',
    'observed_vs_predicted_n_all.png',
    'observed_vs_predicted_n_test.png',
}


def artifact_directory(model_dir):
    root = Path(model_dir)
    return root if (root / 'metrics.json').is_file() else root / 'artifacts'


def saved_model_settings(model_dir, metrics):
    """Use run snapshots only, never today's mutable configuration."""
    settings = dict(metrics.get('model_settings', {}))
    if not settings:
        snapshot = artifact_directory(model_dir) / 'model_config.json'
        if not snapshot.is_file():
            snapshot = Path(model_dir) / 'model_config.json'
        if snapshot.is_file():
            settings = json.loads(snapshot.read_text(encoding='utf-8'))
    for key in ('numeric_features', 'categorical_features', 'training_min_depth_m',
                'training_max_depth_m', 'include_n50'):
        if key in metrics:
            settings[key] = metrics[key]
    if 50 in metrics.get('excluded_observed_n_values', []):
        settings['include_n50'] = False
    return settings


def save_evaluation_table(model_dir, metrics):
    """One row per fold/statistic; all metrics are for capped SPT N-values."""
    root = Path(model_dir)
    rows = []
    cv = metrics.get('cross_validation')
    if cv:
        for fold in cv['folds']:
            rows.append(dict(result=f"fold_{fold['fold']}",
                             **{k: fold[k] for k in ('r2', 'rmse', 'mae')},
                             rows=fold['test_rows'], n_estimators=fold['n_estimators']))
        for statistic in ('mean', 'std'):
            rows.append(dict(result=f'fold_{statistic}',
                             **{k: cv['summary'][k][statistic] for k in ('r2', 'rmse', 'mae')}))
        rows.append(dict(result='cv_oof', **cv['oof'], rows=metrics['rows']['test']))
    elif 'test' in metrics:
        rows.append(dict(result='heldout_test', **metrics['test'], rows=metrics.get('rows', {}).get('test')))
    if 'all_observed' in metrics:
        report = metrics['all_observed']
        rows.append(dict(result='all_observed', **{k: report.get(k) for k in ('r2', 'rmse', 'mae', 'rows')}))
    table = pd.DataFrame(rows)
    settings = saved_model_settings(root, metrics)
    table['include_n50'] = settings.get('include_n50', True)
    table['target'] = 'SPT N-value (capped at 50)'
    table['training_min_depth_m'] = settings.get('training_min_depth_m')
    table['training_max_depth_m'] = settings.get('training_max_depth_m')
    table['std_ddof'] = cv.get('std_ddof', 1) if cv else None
    for key in ('numeric_features', 'categorical_features'):
        table[key] = json.dumps(settings.get(key), ensure_ascii=False)
    table['model_settings_json'] = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    table['validation_config_json'] = json.dumps(metrics.get('validation_config'), ensure_ascii=False, sort_keys=True)
    table['xgb_parameters_json'] = json.dumps(metrics.get('xgb_parameters'), ensure_ascii=False, sort_keys=True)
    table['split_method'] = metrics.get('split_method')
    table['final_fit_rows'] = metrics.get('final_fit_rows')
    table['final_fit_boreholes'] = metrics.get('final_fit_boreholes')
    table['final_n_estimators'] = (metrics['best_iteration'] + 1) if 'best_iteration' in metrics else None
    root.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(root.parent / 'evaluation_metrics.csv', index=False, encoding='utf-8-sig')


def organize_model_output(model_dir):
    """Keep four public plots; preserve all other artifacts under artifacts/."""
    root = Path(model_dir)
    source = artifact_directory(root)
    metadata = source / 'metrics.json'
    if metadata.is_file():
        save_evaluation_table(root, json.loads(metadata.read_text()))
    # Use the importance aggregated over original predictors as the public plot.
    importance = root / 'feature_importance_grouped.png'
    if importance.is_file():
        shutil.copyfile(importance, root / 'feature_importance.png')
    for name in PLOTS:
        if not (root / name).exists() and (source / name).is_file():
            shutil.copyfile(source / name, root / name)
    archive = root / 'artifacts'
    archive.mkdir(parents=True, exist_ok=True)
    for path in list(root.iterdir()):
        if path.name in PLOTS or path == archive:
            continue
        destination = archive / path.name
        if path.is_dir() and destination.exists():
            raise FileExistsError(f'Refusing to overwrite an existing artifact directory: {destination}')
        shutil.move(str(path), str(destination))
