"""Single-process TabPFN runtime setup. Bootstrap uses only the standard library."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import warnings

from module.config_io import load_jsonc_config
from module.paths import project_path

_BOOTSTRAPPED = None


def positive_integer(value, name, nullable=False):
    if value is None and nullable:
        return value
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be an integer >= 1' + (' or null' if nullable else ''))
    return value


def resolve_threads(options):
    requested = positive_integer(options.get('cpu_threads'), 'tabpfn.cpu_threads', True)
    if 'predict_batch_size' in options:
        positive_integer(options['predict_batch_size'], 'tabpfn.predict_batch_size')
    pbs_np = None
    if os.environ.get('PBS_NP'):
        raw = os.environ['PBS_NP']
        if not raw.isdecimal() or int(raw) < 1:
            raise ValueError('PBS_NP must be a positive integer')
        pbs_np = int(raw)
    if not os.environ.get('PBS_NODEFILE'):
        for key in ('PBS_NUM_NODES', 'PBS_NNODES'):
            raw = os.environ.get(key)
            if raw and (not raw.isdecimal() or int(raw) != 1):
                raise ValueError('Without PBS_NODEFILE only a single-node allocation is supported')
    local_slots = None
    nodefile = os.environ.get('PBS_NODEFILE')
    if nodefile:
        try:
            nodes = [v.strip().lower().split('.')[0] for v in Path(nodefile).read_text().splitlines() if v.strip()]
        except OSError as error:
            raise ValueError('Cannot read PBS_NODEFILE; cannot safely establish local allocation') from error
        hosts = {socket.gethostname().lower().split('.')[0], socket.getfqdn().lower().split('.')[0]}
        local_slots = sum(n in hosts for n in nodes)
        if not local_slots:
            raise ValueError('PBS_NODEFILE contains no slots for this host; run on the allocated node')
    limits = [v for v in (pbs_np, local_slots) if v is not None]
    pbs = min(limits) if limits else None
    if os.environ.get('PBS_JOBID') and pbs is None:
        raise ValueError('PBS allocation unknown: provide PBS_NP or a readable PBS_NODEFILE')
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    os_cpus = os.cpu_count() or 1
    available = min(os_cpus, affinity) if affinity else os_cpus
    desired = requested if requested is not None else (pbs or available)
    effective = min(desired, available, pbs or available)
    if desired > effective:
        warnings.warn(f'TabPFN CPU threads capped from {desired} to {effective} by PBS/CPU availability', RuntimeWarning)
    return dict(cpu_threads_configured=requested, cpu_threads=effective, pbs_cpus=pbs,
                pbs_np=pbs_np, pbs_local_slots=local_slots, affinity_cpus=affinity,
                os_cpus=os_cpus, source='config' if requested is not None else ('PBS' if pbs else 'OS'),
                capped=desired != effective)


def prepare_environment(options):
    global _BOOTSTRAPPED
    plan = resolve_threads(options)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(plan['cpu_threads'])
    # No opportunistic downloads when the user later invokes analysis.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
    os.environ['TABPFN_NO_BROWSER'] = '1'
    plan['early_initialization'] = not any(n in sys.modules for n in ('numpy', 'torch', 'sklearn'))
    _BOOTSTRAPPED = plan
    return plan


def apply_runtime(options):
    plan = resolve_threads(options)
    if _BOOTSTRAPPED is None or _BOOTSTRAPPED['cpu_threads'] != plan['cpu_threads']:
        prepare_environment(options)
    plan['early_initialization'] = _BOOTSTRAPPED['early_initialization']
    if not plan['early_initialization']:
        warnings.warn('Numerical libraries already imported; use the repository CLI for early BLAS initialization', RuntimeWarning)
    import torch
    torch.set_num_threads(plan['cpu_threads'])
    plan['torch_threads'] = torch.get_num_threads()
    if plan['torch_threads'] != plan['cpu_threads']:
        raise RuntimeError('PyTorch did not apply the requested thread limit')
    return plan


def batch_size(name, config):
    value = config.get('batch_size', 256)
    if name == 'tabpfn':
        value = config['tabpfn'].get('predict_batch_size', value)
    return positive_integer(value, f'{name}.predict_batch_size')


def runtime_override(config, path):
    if path:
        override = load_jsonc_config(project_path(path))
        if set(override) != {'tabpfn'} or set(override['tabpfn']) - {'cpu_threads', 'device', 'predict_batch_size'}:
            raise ValueError('Runtime override permits only tabpfn.cpu_threads, device, predict_batch_size')
        config['tabpfn'] = {**config['tabpfn'], **override['tabpfn']}
    return config


def bootstrap(argv, entry):
    """Read configuration only, before importing pipeline/NumPy/PyTorch."""
    if not argv or any(v in argv for v in ('--help', '-h', '--plan')):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage', default='all')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--comparison-config', default='config/comparison.json')
    parser.add_argument('--comparison-run')
    parser.add_argument('--output')
    parser.add_argument('--runtime-config')
    args, _ = parser.parse_known_args(argv)
    if entry == 'analysis':
        if args.stage != 'compare' or (args.models is not None and 'tabpfn' not in args.models):
            return
        config = load_jsonc_config(project_path(args.comparison_config))
        if 'tabpfn' not in (args.models if args.models is not None else config['models']):
            return
    else:
        if args.stage not in ('all', 'tabpfn') or not args.comparison_run or not args.output:
            return
        saved = project_path(args.output) / 'config.json'
        config = load_jsonc_config(saved if saved.is_file() else project_path(args.comparison_run) / 'config.json')
        if 'tabpfn' not in config.get('models', []):
            return
        runtime_override(config, args.runtime_config)
    prepare_environment(config['tabpfn'])


def append_log(path, record):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), **record), default=str) + '\n')
