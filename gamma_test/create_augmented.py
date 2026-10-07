"""Create labeled augmentation and evaluate its position-model effect without test leakage."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from module.gamma_features import model_frame
from module.gamma_training import fit_evaluation
from module.validation import split_data
from module.xgb_common import build_preprocessor, build_regressor, regression_metrics
from gamma_test.run_comparison import GROUPS


def main():
    out = Path(__file__).resolve().parent
    config = json.loads((ROOT / 'config/gamma/prediction.json').read_text())
    summary = pd.read_csv(out / 'comparison.csv')
    positions = ['x', 'y', 'elevation']
    all_records = pd.read_csv(out / 'eligible_records.csv')
    all_measured = all_records.loc[all_records.target.notna()].copy()
    for name, lab in GROUPS.items():
        measured = pd.read_csv(out / f'{name}_measured.csv')
        candidates = pd.read_csv(out / f'{name}_candidates.csv')
        features = positions + lab
        for mode in ('borehole', 'spatial'):
            parts = split_data(measured, dict(config['validation'], mode=mode), target_column='target')
            teacher, pre, _ = fit_evaluation(parts, features, [], config['xgb_parameters'])
            heldout = pd.concat(parts[1:])
            eligible = candidates.loc[~candidates.boring_id.isin(heldout.boring_id)].copy()
            if mode == 'spatial':
                import numpy as np
                size = config['validation']['block_size_m']
                blocked = set(zip(*np.floor(heldout[['x', 'y']].to_numpy()/size).astype(int).T))
                blocks = np.floor(eligible[['x', 'y']].to_numpy()/size).astype(int)
                eligible = eligible.loc[[tuple(b) not in blocked for b in blocks]].copy()
            if len(eligible):
                eligible['target'] = teacher.predict(pre.transform(model_frame(eligible, features, [])))
            augmented_parts = (pd.concat([parts[0], eligible], ignore_index=True), *parts[1:])
            _, _, predicted = fit_evaluation(augmented_parts, positions, [], config['xgb_parameters'])
            mask = (summary.dataset == name) & (summary.validation == mode)
            summary.loc[mask, 'augmented_position_r2'] = regression_metrics(parts[2].target, predicted)['r2']
            summary.loc[mask, 'evaluation_added_count'] = len(eligible)
            test = parts[2][['record_id', 'boring_id', 'target']].copy()
            test['augmented_position_prediction'] = predicted
            test.to_csv(out / f'{name}_{mode}_augmented_predictions.csv', index=False)
        # Final augmentation uses all measured records, with tree count selected by validation.
        best = int(summary.loc[(summary.dataset == name) & (summary.validation == 'borehole'), 'best_n_estimators'].iloc[0])
        pre = build_preprocessor(features, [])
        matrix = pre.fit_transform(model_frame(measured, features, []))
        parameters = dict(config['xgb_parameters'], n_estimators=best)
        parameters.pop('early_stopping_rounds', None)
        model = build_regressor(parameters)
        model.fit(matrix, measured.target, verbose=False)
        if len(candidates):
            candidates['target'] = model.predict(pre.transform(model_frame(candidates, features, [])))
        all_measured['label_source'] = 'measured'
        candidates['label_source'] = 'predicted'
        pd.concat([all_measured, candidates], ignore_index=True).to_csv(out / f'{name}_augmented.csv', index=False)
        summary.loc[summary.dataset == name, 'retained_measured_count'] = len(all_measured)
        summary.loc[summary.dataset == name, 'augmented_total_count'] = len(all_measured) + len(candidates)
    summary.to_csv(out / 'comparison.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
