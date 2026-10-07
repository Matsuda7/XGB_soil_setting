"""Date-based output locations, shared for the duration of a pipeline run."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from module.paths import RESULTS

_ACTIVE_DIRECTORY = ContextVar('result_directory', default=None)


def date_directory(root=RESULTS):
    return Path(root) / datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d')


def current_directory():
    return _ACTIVE_DIRECTORY.get() or date_directory()


@contextmanager
def output_session(root):
    directory = date_directory(root)
    token = _ACTIVE_DIRECTORY.set(directory)
    try:
        yield directory
    finally:
        _ACTIVE_DIRECTORY.reset(token)


def create_target_directory(target):
    """Keep previous same-day results before starting a fresh target run."""
    if target not in ('c', 'phi', 'gamma_wet', 'gamma_dry'):
        raise ValueError(f'Unknown output target: {target}')
    directory = current_directory() / target
    if directory.exists():
        archive = directory.parent / 'archive'
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%H%M%S')
        directory.rename(archive / f'{target}_{stamp}_{uuid4().hex[:8]}')
    directory.mkdir(parents=True)
    return directory
