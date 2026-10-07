"""Evaluate B/C/D routing on common holdouts and produce unique pseudo labels."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from module.validation import split_data
from module.gamma_training import fit_evaluation
from module.gamma_features import model_frame
from module.xgb_common import build_preprocessor, build_regressor, regression_metrics

BASE = Path(__file__).resolve().parent
FEATURES = {
    'B': ['x', 'y', 'elevation', 'depth', 'gravel_fraction', 'sand_fraction', 'silt_fraction', 'clay_fraction'],
    'C': ['x', 'y', 'elevation', 'depth', 'water_content'],
}
FEATURES['D'] = FEATURES['B'] + ['water_content']


def available(frame, name):
    return np.isfinite(frame[FEATURES[name]].to_numpy(dtype=float)).all(axis=1)


def main():
    # Keep the original experiment available for paired comparisons and stage 2.
    out = BASE / 'routed_models_depth'
    out.mkdir(exist_ok=True)
    config = json.loads((BASE / 'comparison_settings.json').read_text())['config']
    frame = pd.read_csv(BASE / 'eligible_records.csv')
    for col in FEATURES['D']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    measured = frame.loc[frame.target.notna()].copy()
    candidates = frame.loc[frame.target.isna() & frame.wet_density.isna() &
                           (available(frame, 'B') | available(frame, 'C'))].copy()
    rows, trees = [], {name: [] for name in FEATURES}
    for mode in ('borehole', 'spatial'):
        for seed in (42, 43, 44, 45, 46):
            parts = split_data(measured, dict(config['validation'], mode=mode, random_state=seed), target_column='target')
            test = parts[2].copy()
            predictions = pd.DataFrame(index=test.index)
            for name, features in FEATURES.items():
                subparts = tuple(part.loc[available(part, name)].copy() for part in parts)
                model, _, prediction = fit_evaluation(subparts, features, [], config['xgb_parameters'])
                predictions.loc[subparts[2].index, name] = prediction
                if mode == 'borehole':
                    trees[name].append(model.best_iteration + 1)
            both = predictions.B.notna() & predictions.C.notna()
            for name in FEATURES:
                rows.append(dict(mode=mode, seed=seed, population='both_available', strategy=name,
                                 count=int(both.sum()), **regression_metrics(test.loc[both, 'target'], predictions.loc[both, name])))
            for name, order in {'D_first': ['D', 'C', 'B'], 'C_first': ['C', 'B'], 'B_first': ['B', 'C']}.items():
                prediction = predictions[order].bfill(axis=1).iloc[:, 0]
                valid = prediction.notna()
                rows.append(dict(mode=mode, seed=seed, population='union', strategy=name,
                                 count=int(valid.sum()), **regression_metrics(test.loc[valid, 'target'], prediction.loc[valid])))
                predictions[name] = prediction
            predictions.insert(0, 'target', test.target)
            predictions.insert(0, 'record_id', test.record_id)
            predictions.to_csv(out / f'{mode}_{seed}_predictions.csv', index=False)
            pd.concat([part[['record_id', 'boring_id']].assign(split=label)
                       for label, part in zip(('train', 'validation', 'test'), parts)]).to_csv(
                           out / f'{mode}_{seed}_split.csv', index=False)
            print(f'Completed {mode} seed={seed}', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'scores.csv', index=False)
    summary = scores.groupby(['mode', 'population', 'strategy']).agg(
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'), r2_min=('r2', 'min'),
        r2_max=('r2', 'max'), rmse_mean=('rmse', 'mean'), test_count_mean=('count', 'mean'))
    summary.to_csv(out / 'summary.csv')
    candidates['source_model'] = np.where(available(candidates, 'D'), 'D', np.where(available(candidates, 'C'), 'C', 'B'))
    for name, features in FEATURES.items():
        data = measured.loc[available(measured, name)]
        selected = candidates.source_model == name
        if not selected.any():
            continue
        pre = build_preprocessor(features, [])
        x = pre.fit_transform(model_frame(data, features, []))
        parameters = dict(config['xgb_parameters'], n_estimators=int(np.median(trees[name])))
        parameters.pop('early_stopping_rounds', None)
        model = build_regressor(parameters)
        model.fit(x, data.target, verbose=False)
        candidates.loc[selected, 'target'] = model.predict(pre.transform(model_frame(candidates.loc[selected], features, [])))
    candidates['label_source'] = 'predicted'
    measured['label_source'] = 'measured'
    combined = pd.concat([measured, candidates], ignore_index=True)
    assert not combined.record_id.duplicated().any()
    assert np.isfinite(combined.target).all() and (combined.target > 0).all()
    candidates.to_csv(out / 'pseudo_labels.csv', index=False)
    combined.to_csv(out / 'augmented.csv', index=False)
    counts = dict(measured=len(measured), predicted=len(candidates), total=len(combined),
                  by_model=candidates.source_model.value_counts().to_dict(),
                  training_counts={name: int(available(measured, name).sum()) for name in FEATURES})
    (out / 'settings.json').write_text(json.dumps(dict(config=config, features=FEATURES,
        split_seeds=[42,43,44,45,46], final_tree_counts={k:int(np.median(v)) for k,v in trees.items()},
        counts=counts), indent=2)+'\n')
    print(summary.to_string())
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
