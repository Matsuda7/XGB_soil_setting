"""Reproducible spatial comparison, separate from legacy analysis outputs."""
from datetime import datetime
from zoneinfo import ZoneInfo
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import os
import sys
import subprocess
import uuid
import joblib
import numpy as np
import pandas as pd
from module.paths import project_path, RESULTS, ROOT
from module.xgb_common import build_preprocessor, load_jsonc_config
from module.gamma_features import model_frame
from module.comparison_data import load_target, spatial_folds, TARGETS
from module.comparison_models import build_model, predict_batches, save_model
from module.tabpfn_runtime import batch_size
from module.comparison_reporting import metrics, plots


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'torch', 'tabpfn', 'joblib'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
                packages=packages, git_sha=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                timing='wall seconds; fit includes model initialization; predict includes lazy TabPFN work')


def dense(values):
    return np.asarray(values.toarray() if hasattr(values, 'toarray') else values, dtype=np.float32)


def evaluate_target(name, spec, config, output):
    data, excluded, target, unit = load_target(name, spec, config)
    directory = output / name
    directory.mkdir(parents=True)
    splits = spatial_folds(data, config)
    splits.to_csv(directory / 'cv_splits.csv', index=False)
    excluded.to_csv(directory / 'excluded_rows.csv', index=False)
    numeric, categorical = spec['numeric_features'], spec['categorical_features']
    features = model_frame(data, numeric, categorical)
    columns = list(dict.fromkeys(['row_id', 'source_row_id', 'boring_id', 'x', 'y', 'depth', target, *numeric, *categorical]))
    data[columns].to_csv(directory / 'training_snapshot.csv.gz', index=False)
    metadata = dict(target=target, unit=unit, rows=len(data), excluded_rows=len(excluded),
                    source_sha256=sha256(project_path(spec['data'])), features=spec,
                    policy='measured targets; geographic features; no pseudo labels; fixed hyperparameters',
                    evaluation='raw predictions; no clipping', splits_sha256=sha256(directory / 'cv_splits.csv'))
    metadata['snapshot_sha256'] = sha256(directory / 'training_snapshot.csv.gz')
    write_json(directory / 'dataset.json', metadata)
    oof, scores, errors = [], [], []
    active = set(config['models'])
    for fold in sorted(splits.fold.unique()):
        train = splits.fold.to_numpy() != fold
        test = ~train
        pre = build_preprocessor(numeric, categorical)
        started = perf_counter()
        xt = dense(pre.fit_transform(features.loc[train]))
        xv = dense(pre.transform(features.loc[test]))
        preprocessing_seconds = perf_counter() - started
        for model_name in config['models']:
            if model_name not in active:
                continue
            dest = directory / model_name / f'fold_{fold}'
            dest.mkdir(parents=True)
            try:
                started = perf_counter()
                model = build_model(model_name, config, len(xt), xt.shape[1])
                model.fit(xt, data.loc[train, target].to_numpy())
                fit_seconds = perf_counter() - started
                started = perf_counter()
                pred = predict_batches(model, xv, batch_size(model_name, config),
                                       dest / 'prediction_runtime.jsonl' if model_name == 'tabpfn' else None)
                predict_seconds = perf_counter() - started
                table = splits.loc[test].copy()
                table['target'] = name; table['model'] = model_name
                table['actual'] = data.loc[test, target].to_numpy()
                table['predicted'] = pred; table['residual'] = pred - table.actual
                if name == 'N':
                    from module.phi_prediction import PHI_MAX_DEG
                    table['predicted_phi_deg'] = np.minimum(np.sqrt(15 * np.clip(pred, 0, 50)) + 15, PHI_MAX_DEG)
                    table['phi_reference_type'] = 'conversion_only_no_observed_phi_evaluation'
                score = dict(target=name, model=model_name, fold=int(fold),
                             **metrics(table.actual, pred), train_rows=int(train.sum()), test_rows=int(test.sum()),
                             encoded_features=xt.shape[1], preprocessing_seconds=preprocessing_seconds,
                             fit_seconds=fit_seconds, predict_seconds=predict_seconds)
                save_model(model, model_name, dest)
                joblib.dump(pre, dest / 'preprocessor.joblib')
                write_json(dest / 'parameters.json', model.get_params())
                table.to_csv(dest / 'predictions.csv', index=False)
                oof.append(table); scores.append(score)
                print(f'{name} {model_name} fold {fold}: {score["rmse"]:.4f} RMSE', flush=True)
                del model
            except Exception as error:
                failure = dict(target=name, model=model_name, fold=int(fold), error=f'{type(error).__name__}: {error}')
                errors.append(failure); active.remove(model_name)
                write_json(dest / 'failure.json', failure)
                print(f'Unavailable: {failure}', flush=True)
    all_oof = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
    by_fold = pd.DataFrame(scores)
    summary = []
    for model_name in config['models']:
        if all_oof.empty:
            continue
        rows = all_oof[all_oof.model == model_name]
        complete = len(rows) == len(data) and rows.row_id.nunique() == len(data)
        if not complete:
            continue  # Never label partial folds as a full OOF result.
        fold_rows = by_fold[by_fold.model == model_name]
        summary.append(dict(target=name, model=model_name, population='OOF', unit=unit,
                            data_kind=config.get('data_kind', 'observed'),
                            **metrics(rows.actual, rows.predicted), rows=len(rows),
                            mean_fold_r2=fold_rows.r2.mean(), std_fold_r2=fold_rows.r2.std(ddof=1),
                            mean_fold_rmse=fold_rows.rmse.mean(), mean_fold_mae=fold_rows.mae.mean()))
    all_oof.to_csv(directory / 'oof_predictions.csv', index=False)
    by_fold.to_csv(directory / 'metrics_by_fold.csv', index=False)
    pd.DataFrame(summary).to_csv(directory / 'metrics_summary.csv', index=False)
    completed = [s['model'] for s in summary]
    if completed and config.get('save_plots', True):
        plots(all_oof[all_oof.model.isin(completed)], by_fold[by_fold.model.isin(completed)], directory, unit)
    write_json(directory / 'status.json', dict(complete_models=completed, errors=errors))
    return summary, by_fold, all_oof, errors


def run_comparison(config_path, models=None, targets=None):
    config = load_jsonc_config(project_path(config_path))
    if models is not None:
        config['models'] = models
    if targets is not None:
        config['targets'] = {k: config['targets'][k] for k in targets}
    selected = config['models']
    if config.get('data_kind', 'observed') not in ('observed', 'synthetic'):
        raise ValueError('data_kind must be observed or synthetic')
    if not selected or len(set(selected)) != len(selected) or set(selected) - {'xgboost', 'tabpfn'}:
        raise ValueError('models must contain xgboost and/or tabpfn without duplicates')
    if not config['targets'] or set(config['targets']) - set(TARGETS):
        raise ValueError('Unknown or empty comparison targets')
    now = datetime.now(ZoneInfo('Asia/Tokyo'))
    output = RESULTS / now.strftime('%Y%m%d') / 'comparison' / (now.strftime('%H%M%S') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'run_summary.json', dict(status='running', started_at=now.isoformat(),
               data_kind=config.get('data_kind', 'observed'), output=str(output)))
    write_json(output / 'config.json', config)
    info = environment()
    info['started_at'] = now.isoformat()
    info['source_files_sha256'] = {str(p.relative_to(ROOT)): sha256(p)
                                    for pattern in ('module/*.py', 'config/comparison.json', 'requirements*.txt')
                                    for p in ROOT.glob(pattern)}
    info['thread_environment'] = {k: os.environ.get(k)
                                   for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    info['xgboost_device'] = config['xgboost'].get('device', 'cpu')
    if 'tabpfn' in selected:
        try:
            from module.comparison_models import tabpfn_device
            info['tabpfn_device'] = tabpfn_device(config['tabpfn'])
            if info['tabpfn_device'] == 'cuda':
                import torch
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
        except (ImportError, RuntimeError, ValueError) as error:
            info['tabpfn_device'] = f'unavailable: {error}'
    checkpoint = project_path(config['tabpfn']['model_path'])
    if checkpoint.is_file():
        info['tabpfn_checkpoint_sha256'] = sha256(checkpoint)
    info['tabpfn_model_family'] = config['tabpfn'].get('model_version', 'v2')
    write_json(output / 'environment.json', info)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True)
    (output / 'environment_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    summary, folds, predictions, errors = [], [], [], []
    for name, spec in config['targets'].items():
        try:
            s, f, p, e = evaluate_target(name, spec, config, output)
            summary.extend(s); folds.append(f); predictions.append(p); errors.extend(e)
        except Exception as error:
            errors.append(dict(target=name, error=f'{type(error).__name__}: {error}'))
            print(errors[-1], flush=True)
    pd.DataFrame(summary).to_csv(output / 'metrics_summary.csv', index=False)
    (pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()).to_csv(output / 'metrics_by_fold.csv', index=False)
    (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()).to_csv(output / 'oof_predictions.csv', index=False)
    # Distribution follows completed CV only, and uses a shared feature grid.
    if not errors and config.get('distribution', {}).get('enabled', False):
        from module.comparison_distribution import distribute
        try:
            distribute(config, output)
        except Exception as error:
            errors.append(dict(stage='distribution', error=f'{type(error).__name__}: {error}'))
    status = dict(status='incomplete' if errors else 'complete', errors=errors,
                  data_kind=config.get('data_kind', 'observed'),
                  completed_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), output=str(output))
    write_json(output / 'run_summary.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 2 if errors else 0
