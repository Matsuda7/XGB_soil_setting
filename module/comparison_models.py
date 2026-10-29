"""Optional model adapters; XGBoost never imports torch or TabPFN."""
from importlib.metadata import version
from pathlib import Path
import os
import json
import zipfile
from time import perf_counter
import numpy as np
from module.xgb_common import build_regressor
from module.paths import project_path
from module.tabpfn_runtime import apply_runtime, append_log

def tabpfn_device(options):
    import torch
    device = options.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda'):
        raise ValueError('TabPFN device must be auto, cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def check_capacity(rows, columns, options):
    device = tabpfn_device(options)
    if options.get('model_version', 'v2') == 'v3.5-fast':
        # The installed official API additionally validates checkpoint-specific CPU limits.
        if rows > 1000000 or columns > 20000:
            raise ValueError('TabPFN 3.5-Fast capacity exceeded; no subsampling performed')
        return device
    limit = 1000 if device == 'cpu' else 10000
    if rows > limit or columns > 500:
        raise ValueError(f'TabPFN v2 capacity: {rows} training rows, {columns} encoded features; '
                         f'{device} limit {limit} rows / 500 features. No subsampling performed. '
                         'Use GPU within v2 limits or validate a newer model family separately.')
    return device


def initialize_tabpfn(options):
    family = options.get('model_version', 'v2')
    if family not in ('v2', 'v3.5-fast'):
        raise ValueError('tabpfn.model_version must be v2 or v3.5-fast')
    expected = '2.2.1' if family == 'v2' else '9.2.0'
    installed = version('tabpfn')
    if installed != expected:
        raise RuntimeError(f'{family} requires tabpfn=={expected}; found {installed}. Use a separate environment.')
    if options.get('n_jobs', 1) != 1:
        raise ValueError('TabPFN preprocessing n_jobs must be 1; use cpu_threads for PyTorch parallelism')
    path = project_path(options['model_path']).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Local {family} checkpoint required at {path}; no automatic download')
    if family == 'v3.5-fast' and path.name != 'tabpfn-v3.5-fast-20260909.safetensors':
        raise ValueError('Use the official 3.5-Fast checkpoint with its original filename')
    runtime = apply_runtime(options)
    runtime.update(device=tabpfn_device(options), model_version=family, tabpfn_version=installed,
                   model_path=options.get('model_path'))
    print('TabPFN runtime:', runtime, flush=True)
    return runtime


def validate_saved_checkpoint(archive, options):
    """Prevent fitted-state restoration from downloading missing original weights."""
    with zipfile.ZipFile(archive) as saved:
        params = json.loads(saved.read('init_params.json'))
    original = params.get('model_path')
    if not isinstance(original, str) or original == 'auto':
        raise ValueError('Saved TabPFN state must reference an explicit local checkpoint')
    original = Path(original).expanduser().resolve()
    if not original.is_file() or original != project_path(options['model_path']).resolve():
        raise ValueError('Saved checkpoint path is missing or differs from configuration; restore the original local weights before resuming')


def build_model(name, config, rows, columns):
    seed = config['random_state']
    if name == 'xgboost':
        params = dict(config['xgboost'])
        if 'early_stopping_rounds' in params or 'callbacks' in params:
            raise ValueError('Comparison uses preregistered fixed XGBoost parameters; no outer-fold tuning')
        params['random_state'] = seed
        return build_regressor(params)
    if name != 'tabpfn':
        raise ValueError(f'Unknown model: {name}')
    options = config['tabpfn']
    runtime = initialize_tabpfn(options)
    device = check_capacity(rows, columns, options)
    os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
    from tabpfn import TabPFNRegressor
    path = project_path(options['model_path']).resolve()
    kwargs = dict(model_path=str(path), device=device,
                          n_estimators=options.get('n_estimators', 8),
                          random_state=seed,
                          fit_mode='low_memory', memory_saving_mode=True,
                          ignore_pretraining_limits=False)
    if runtime['model_version'] == 'v3.5-fast':
        from tabpfn.constants import ModelVersion
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5_FAST, n_preprocessing_jobs=1, **kwargs)
    else:
        model = TabPFNRegressor(n_jobs=1, **kwargs)
    model._soil_runtime = runtime
    return model


def predict_batches(model, values, batch_size, log_path=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    result = []
    if hasattr(model, '_soil_runtime'):
        import torch
        if torch.get_num_threads() != model._soil_runtime['cpu_threads']:
            torch.set_num_threads(model._soil_runtime['cpu_threads'])
        model._soil_runtime['torch_threads'] = torch.get_num_threads()
    started = perf_counter()
    for start in range(0, len(values), batch_size):
        try:
            result.append(np.asarray(model.predict(values[start:start + batch_size]), dtype=float))
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError('Prediction OOM: reduce batch_size or use more memory; '
                                   'training rows were not reduced') from error
            raise
    predicted = np.concatenate(result) if result else np.empty(0)
    if not np.isfinite(predicted).all():
        raise ValueError('Non-finite predictions')
    if hasattr(model, '_soil_runtime'):
        seconds = perf_counter() - started
        append_log(log_path, dict(**model._soil_runtime, batch_size=batch_size,
                                 prediction_points=len(values), inference_seconds=seconds,
                                 points_per_second=len(values) / max(seconds, 1e-9)))
    return predicted


def save_model(model, name, directory):
    if name == 'xgboost':
        model.save_model(directory / 'model.json')
    else:
        model.save_fit_state(directory / 'model.tabpfn_fit')
        append_log(directory / 'runtime.jsonl', getattr(model, '_soil_runtime', {}))
