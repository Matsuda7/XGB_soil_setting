"""Collect existing raw data once, and optionally download soil-test XML.

Hard links avoid copying file contents. Raw files must be treated as immutable;
replace a file atomically when refreshing it. Existing imports are never silently
replaced, and source project directories are not modified.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from module.paths import ROOT, RAW, CACHE, LOGS, project_path


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def import_file(source: Path, destination: Path) -> dict:
    if destination.is_file():
        if source.is_file() and not os.path.samefile(source, destination):
            if fingerprint(source) != fingerprint(destination):
                raise ValueError(f'Import conflict; existing data preserved: {destination}')
        return {'path': str(destination.relative_to(ROOT)), 'status': 'existing'}
    if not source.is_file():
        return {'path': str(destination.relative_to(ROOT)), 'status': 'missing', 'source': str(source)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Do not fall back to a large copy or an external directory symlink.
    os.link(source, destination)
    return {'path': str(destination.relative_to(ROOT)), 'status': 'imported',
            'source': str(source), 'bytes': destination.stat().st_size,
            'sha256': fingerprint(destination), 'method': 'hardlink'}


def collect(config: dict, download: bool = False) -> dict:
    records = []
    for item in config['sources']:
        source = project_path(item['source'])
        destination = project_path(item['destination'])
        if item.get('directory'):
            files = sorted(p for p in source.rglob('*') if p.is_file() and p.name != '.gitkeep') if source.is_dir() else []
            if not files and not any(p.is_file() and p.name != '.gitkeep' for p in destination.rglob('*')):
                records.append({'path': str(destination.relative_to(ROOT)), 'source': str(source), 'status': 'missing'})
            for path in files:
                records.append(import_file(path, destination / path.relative_to(source)))
        else:
            records.append(import_file(source, destination))
    if download:
        from module.download import main
        main(['--output-dir', str(RAW / 'soiltest/strength'),
              '--metadata-dir', str(RAW / 'soiltest/metadata'), '--log-dir', str(LOGS),
              '--bbox', config['soiltest_bbox']])
    manifest_path = RAW / 'collection_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {'files': []}
    manifest = {item['path']: item for item in previous['files']}
    for record in records:
        manifest[record['path']] = {**manifest.get(record['path'], {}), **record}
    RAW.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'files': list(manifest.values())}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    return {'files': records}


def missing_inputs(target: str, stage: str) -> list[str]:
    if target in {'c', 'gamma'}:
        return [] if any((RAW / 'soiltest').rglob('*.xml')) else ['data/raw/soiltest/**/*.xml']
    required = [RAW / 'boring/BorToCsv.csv', RAW / 'dem/z.txt']
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    for folder, pattern in [('geology', '*.shp'), ('jshis', 'Z-V4-JAPAN-AMP-VS400_M250.csv')]:
        if not any((RAW / folder).rglob(pattern)):
            missing.append(f'data/raw/{folder}/**/{pattern}')
    if stage == 'distribute':
        config = json.loads((ROOT / 'config/prediction_phi.json').read_text())
        for key in ('grid_input', 'property_mask'):
            if config.get(key) is not None and not project_path(config[key]).is_file():
                missing.append(str(config[key]))
    return missing
