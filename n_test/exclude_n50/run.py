"""Run phi's N-value XGBoost pipeline after excluding capped N=50 records."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from module import phi_training
from module.paths import RESULTS
from module.xgb_common import load_jsonc_config
from module.phi_output import organize_model_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS / datetime.now().strftime('%Y%m%d') / 'phi_exclude_n50')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh experiment output directory: {output}')
    output.mkdir(parents=True)
    inputs = output / 'input'
    inputs.mkdir()
    source = phi_training.INPUT_CSV
    original = pd.read_csv(source)
    selected = original.loc[original.n_value.ne(50)].copy()
    assert selected.n_value.lt(50).all()
    dataset = inputs / 'model_dataset.csv'
    selected.to_csv(dataset, index=False, encoding='utf-8-sig')
    settings = load_jsonc_config(phi_training.MODEL_CONFIG)
    settings['include_n50'] = False
    config = inputs / 'model.json'
    config.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    lower, upper = settings.get('training_min_depth_m'), settings.get('training_max_depth_m')
    in_depth = pd.Series(True, index=original.index)
    if lower is not None or upper is not None:
        in_depth = original.depth.between(lower if lower is not None else 0, upper if upper is not None else float('inf'))
    report = dict(experiment='Exclude observed N=50 before depth filtering and 5-fold CV',
                  original_dataset=str(source), experiment_dataset=str(dataset),
                  original_rows=len(original), excluded_n50_rows=int(original.n_value.eq(50).sum()),
                  rows_before_exclusion_in_depth=int(in_depth.sum()),
                  excluded_n50_rows_in_depth=int((in_depth & original.n_value.eq(50)).sum()),
                  retained_rows_in_depth=int((in_depth & original.n_value.ne(50)).sum()),
                  model_settings=settings, status='running')
    report_path = output / 'experiment.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    old_input, old_config = phi_training.INPUT_CSV, phi_training.MODEL_CONFIG
    try:
        phi_training.INPUT_CSV, phi_training.MODEL_CONFIG = dataset, config
        phi_training.main(output / 'model')
        metadata = output / 'model/artifacts/metrics.json'
        metrics = json.loads(metadata.read_text())
        metrics['training_dataset'] = str(dataset)
        metrics['excluded_observed_n_values'] = [50]
        metrics['experiment'] = report['experiment']
        metadata.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
        organize_model_output(output / 'model')
        for filename in ('test_predictions.csv', 'all_observed_predictions.csv'):
            predictions = pd.read_csv(output / 'model/artifacts' / filename)
            assert predictions.n_value.lt(50).all()
            assert len(predictions) == report['retained_rows_in_depth']
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        phi_training.INPUT_CSV, phi_training.MODEL_CONFIG = old_input, old_config
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print((output / 'evaluation_metrics.csv').read_text(encoding='utf-8-sig'), flush=True)


if __name__ == '__main__':
    main()
