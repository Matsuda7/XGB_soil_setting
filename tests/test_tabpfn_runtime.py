"""Runtime-only test definitions. No learning, inference, weights or data needed.

Written but deliberately NOT executed for the CPU-thread implementation task.
"""
import sys
from types import SimpleNamespace
import pytest
from module import tabpfn_runtime as runtime


@pytest.fixture
def host(monkeypatch):
    for key in ('PBS_NP', 'PBS_NODEFILE', 'PBS_JOBID', 'PBS_NUM_NODES', 'PBS_NNODES'):
        monkeypatch.delenv(key, raising=False)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'HF_HUB_OFFLINE', 'TABPFN_DISABLE_TELEMETRY', 'TABPFN_NO_BROWSER'):
        monkeypatch.setenv(key, runtime.os.environ.get(key, ''))
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 16)
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda _: set(range(12)), raising=False)
    monkeypatch.setattr(runtime.socket, 'gethostname', lambda: 'orca01')
    monkeypatch.setattr(runtime.socket, 'getfqdn', lambda: 'orca01.local')
    monkeypatch.setattr(runtime, '_BOOTSTRAPPED', None)


@pytest.mark.parametrize('value', [0, -1, 1.5, '8', True, False, [], {}])
def test_invalid_threads(host, value):
    with pytest.raises(ValueError, match='cpu_threads'):
        runtime.resolve_threads({'cpu_threads': value})


def test_auto_affinity_and_explicit(host):
    assert runtime.resolve_threads({})['cpu_threads'] == 12
    assert runtime.resolve_threads({'cpu_threads': 3})['cpu_threads'] == 3


def test_pbs_cap(host, monkeypatch):
    monkeypatch.setenv('PBS_NP', '8')
    assert runtime.resolve_threads({})['cpu_threads'] == 8
    with pytest.warns(RuntimeWarning):
        assert runtime.resolve_threads({'cpu_threads': 20})['cpu_threads'] == 8


def test_local_slots_not_cluster_total(host, monkeypatch, tmp_path):
    nodes = tmp_path/'nodes'
    nodes.write_text('orca01.local\n'*4 + 'orca02\n'*8)
    monkeypatch.setenv('PBS_NODEFILE', str(nodes))
    monkeypatch.setenv('PBS_NP', '12')
    assert runtime.resolve_threads({})['cpu_threads'] == 4


def test_unknown_allocation_fails_closed(host, monkeypatch):
    monkeypatch.setenv('PBS_JOBID', '123.orca')
    with pytest.raises(ValueError, match='allocation unknown'):
        runtime.resolve_threads({})


@pytest.mark.parametrize('value', [None, 0, -2, 1.2, '1000', True])
def test_invalid_batch(host, value):
    with pytest.raises(ValueError, match='predict_batch_size'):
        runtime.resolve_threads({'predict_batch_size': value})


def test_pytorch_api_with_fake_module(host, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda n: calls.append(n), get_num_threads=lambda: calls[-1]))
    plan = runtime.apply_runtime({'cpu_threads': 4})
    assert calls == [4]
    assert plan['torch_threads'] == 4
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert runtime.os.environ[name] == '4'


def test_xgboost_bootstrap_is_noop(host, monkeypatch):
    monkeypatch.setattr(runtime, 'prepare_environment', lambda _: pytest.fail('TabPFN initialized'))
    runtime.bootstrap(['--stage', 'compare', '--models', 'xgboost'], 'analysis')
    runtime.bootstrap(['--stage', 'xgboost'], 'distribution')
    runtime.bootstrap(['--stage', 'compare', '--plan'], 'analysis')


def test_pbs_affinity_cap(host, monkeypatch):
    monkeypatch.setenv('PBS_NP', '32')
    with pytest.warns(RuntimeWarning):
        plan = runtime.resolve_threads({'cpu_threads': None})
    assert plan['cpu_threads'] == 12
    assert plan['pbs_cpus'] == 32


def test_missing_nodefile_fails_closed(host, monkeypatch, tmp_path):
    monkeypatch.setenv('PBS_NP', '8')
    monkeypatch.setenv('PBS_NODEFILE', str(tmp_path/'missing'))
    with pytest.raises(ValueError, match='Cannot read'):
        runtime.resolve_threads({})


def test_config_batch_priority(host):
    config = {'batch_size': 256, 'tabpfn': {'predict_batch_size': 1000}}
    assert runtime.batch_size('tabpfn', config) == 1000
    assert runtime.batch_size('xgboost', config) == 256


def test_runtime_override_rejects_model_change(host, tmp_path):
    path = tmp_path/'override.json'
    path.write_text('{"tabpfn":{"model_version":"v3.5-fast"}}')
    with pytest.raises(ValueError, match='Runtime override'):
        runtime.runtime_override({'tabpfn': {}}, path)
