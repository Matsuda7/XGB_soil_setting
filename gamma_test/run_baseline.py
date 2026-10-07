"""Evaluate the existing wet-gamma data; additional datasets are pending specification."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from module.gamma_training import fit_evaluation
from module.validation import split_data
from module.xgb_common import regression_metrics


def main():
    output = Path(__file__).resolve().parent
    config = json.loads((ROOT / 'config/gamma/prediction.json').read_text())
    data = pd.read_csv(ROOT / 'data/training/gamma/wet.csv')
    if 'elevation' not in data:
        data = data.rename(columns={'sample_z': 'elevation'})
    features = ['x', 'y', 'elevation']
    rows = []
    for mode in ('borehole', 'spatial'):
        settings = dict(config['validation'], mode=mode)
        parts = split_data(data, settings, target_column='target')
        model, _, prediction = fit_evaluation(parts, features, [], config['xgb_parameters'])
        metrics = regression_metrics(parts[2].target, prediction)
        rows.append(dict(dataset='existing_wet_baseline', validation=mode,
                         wet_count=len(data), borehole_count=data.boring_id.nunique(),
                         train_count=len(parts[0]), validation_count=len(parts[1]),
                         test_count=len(parts[2]), best_n_estimators=model.best_iteration + 1,
                         **metrics))
        test = parts[2][['record_id', 'boring_id', *features, 'target']].copy()
        test['predicted_gamma_wet'] = prediction
        test.to_csv(output / f'baseline_{mode}_predictions.csv', index=False)
        pd.concat([part[['record_id', 'boring_id']].assign(split=name)
                   for name, part in zip(('train', 'validation', 'test'), parts)]).to_csv(
                       output / f'baseline_{mode}_split.csv', index=False)
    pd.DataFrame(rows).to_csv(output / 'baseline_summary.csv', index=False)
    (output / 'baseline_settings.json').write_text(json.dumps(
        dict(source='data/training/gamma/wet.csv', numeric_features=features,
             xgb_parameters=config['xgb_parameters'], validation=config['validation'],
             status='baseline_only_additional_datasets_pending'), indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
