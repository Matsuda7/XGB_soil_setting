"""Sequential three-stage orchestration and explicit per-target status reports."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import traceback
from module.paths import ROOT, CONFIG, RESULTS, project_path
from module.collection import collect, missing_inputs

STAGES = ('collect', 'prepare', 'distribute')
TARGETS = ('c', 'phi', 'gamma')


def resolve_targets(config, override=None):
    """Prefer an explicit override, then configured targets, then legacy defaults."""
    selected = override if override is not None else config.get('targets', list(TARGETS))
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError('targets must be a non-empty list, for example ["phi", "gamma"]')
    if any(not isinstance(target, str) or target not in TARGETS for target in selected):
        raise ValueError('targets may only contain: c, phi, gamma')
    return tuple(dict.fromkeys(selected))


def run(stage='all', targets=None, config_path=CONFIG / 'pipeline.json', download=False):
    config = json.loads(project_path(config_path).read_text(encoding='utf-8'))
    targets = resolve_targets(config, targets)
    stages = STAGES if stage == 'all' else (stage,)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'stages': {}, 'targets': list(targets)}
    failed = False
    collection_failed = False
    preparation = {}
    for current in stages:
        print(f'[{current}] {", ".join(targets)}', flush=True)
        entries = {}
        if collection_failed and current != 'collect':
            report['stages'][current] = {t: {'status': 'blocked', 'reason': 'Data collection failed'} for t in targets}
            continue
        if current == 'collect':
            try:
                entries = collect(config, download)
                entries['targets'] = {t: {'missing': missing_inputs(t, 'distribute')} for t in targets}
                if any(row['missing'] for row in entries['targets'].values()):
                    failed = True
            except Exception as error:
                entries = {'status': 'failed', 'error': str(error)}
                failed = True
                collection_failed = True
                traceback.print_exc()
        elif current == 'prepare':
            from module.preparation import prepare_strength, prepare_phi
            ready = []
            for target in targets:
                missing = missing_inputs(target, current)
                if missing:
                    entries[target] = {'status': 'missing_input', 'missing': missing}
                    failed = True
                elif target in ('c', 'gamma'):
                    ready.append(target)
                else:
                    try:
                        entries[target] = prepare_phi()
                    except Exception as error:
                        entries[target] = {'status': 'failed', 'error': str(error)}
                        failed = True
                        traceback.print_exc()
            if ready:
                try:
                    entries.update(prepare_strength(ready))
                except Exception as error:
                    entries.update({t: {'status': 'failed', 'error': str(error)} for t in ready})
                    failed = True
                    traceback.print_exc()
            preparation = entries
            failed |= any(v['status'] != 'prepared' for v in entries.values())
        else:
            from module.distributions import distribute
            for target in targets:
                missing = missing_inputs(target, current)
                if missing:
                    entries[target] = {'status': 'missing_input', 'missing': missing}
                elif target in preparation and preparation[target]['status'] != 'prepared':
                    entries[target] = {'status': 'blocked', 'reason': 'Preparation did not succeed'}
                else:
                    try:
                        entries[target] = distribute(target)
                    except Exception as error:
                        entries[target] = {'status': 'failed', 'error': str(error)}
                        traceback.print_exc()
                failed |= entries[target]['status'] != 'complete'
        report['stages'][current] = entries
        if current == 'collect':
            for target, entry in entries.get('targets', {}).items():
                print(f"  {target}: " + (f"missing {', '.join(entry['missing'])}" if entry['missing'] else 'inputs available'), flush=True)
        else:
            for target, entry in entries.items():
                print(f"  {target}: {entry['status']}", flush=True)
    report['status'] = 'incomplete' if failed else 'complete'
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / 'run_summary.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)
    print(f"Status: {report['status']}; report: {destination}")
    return 2 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', *STAGES], default='all')
    parser.add_argument('--targets', choices=TARGETS, nargs='+', default=None, help='Override config targets (c, phi, gamma)')
    parser.add_argument('--config', default=str(CONFIG / 'pipeline.json'))
    parser.add_argument('--download', action='store_true', help='Refresh soil-test XML from KuniJiban during collection')
    parser.add_argument('--plan', action='store_true', help='Show stages and missing inputs without executing analysis')
    args = parser.parse_args(argv)
    try:
        config = json.loads(project_path(args.config).read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ValueError('Pipeline config must be a JSON object')
        targets = resolve_targets(config, args.targets)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps({'stages': STAGES if args.stage == 'all' else [args.stage],
                          'targets': {t: {'missing': missing_inputs(t, 'distribute'),
                                          'distribution_implemented': t in ('c', 'phi', 'gamma')} for t in targets}},
                         ensure_ascii=False, indent=2))
        return 0
    return run(args.stage, targets, args.config, args.download)
