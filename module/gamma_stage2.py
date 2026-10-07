"""Isolated two-stage gamma baseline with nested, measured-only evaluation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from module.paths import project_path
from module.validation import split_data
from module.gamma_training import fit_evaluation
from module.gamma_features import model_frame
from module.xgb_common import build_preprocessor, build_regressor, regression_metrics


def exclude_heldout(candidates, heldout):
    keep = ~candidates.boring_id.isin(heldout.boring_id)
    held_xy = set(zip(heldout.x, heldout.y))
    keep &= pd.Series([xy not in held_xy for xy in zip(candidates.x, candidates.y)], index=candidates.index)
    return candidates.loc[keep].copy()


def fit_final(data, numeric, categorical, parameters, trees):
    pre = build_preprocessor(numeric, categorical)
    x = pre.fit_transform(model_frame(data, numeric, categorical))
    params = dict(parameters, n_estimators=int(trees))
    params.pop('early_stopping_rounds', None)
    model = build_regressor(params)
    model.fit(x, data.target, verbose=False)
    return model, pre


def regenerate(train, candidates, features, params, config, seed, output):
    """Teacher tuning and refitting use only the outer training measurements."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=config['stage1_inner_validation_size'], random_state=seed+101)
    a, b = next(splitter.split(train, groups=train._validation_group))
    inner = (train.iloc[a], train.iloc[b])
    pd.concat([part[['record_id', 'boring_id']].assign(split=label)
               for label, part in zip(('teacher_train', 'teacher_validation'), inner)]).to_csv(output/'teacher_split.csv', index=False)
    pseudo = candidates.copy()
    available = {name: np.isfinite(pseudo[cols].to_numpy(float)).all(axis=1) for name, cols in features.items()}
    pseudo['source_model'] = np.where(available['D'], 'D', np.where(available['C'], 'C', 'B'))
    pseudo['target'] = np.nan  # Never reuse all-data pseudo labels during evaluation.
    metadata = {}
    for name, cols in features.items():
        subset = tuple(p.loc[np.isfinite(p[cols].to_numpy(float)).all(axis=1)] for p in inner)
        if min(map(len, subset)) < 2:
            raise ValueError(f'Insufficient inner data for teacher {name}')
        model, _, _ = fit_evaluation((*subset, subset[1]), cols, [], params)
        trees = model.best_iteration + 1
        full = train.loc[np.isfinite(train[cols].to_numpy(float)).all(axis=1)]
        model, pre = fit_final(full, cols, [], params, trees)
        selected = pseudo.source_model == name
        if selected.any():
            pseudo.loc[selected, 'target'] = model.predict(pre.transform(model_frame(pseudo.loc[selected], cols, [])))
        metadata[name] = dict(refit_rows=len(full), inner_train=len(subset[0]), inner_validation=len(subset[1]), trees=trees)
    if not np.isfinite(pseudo.target).all():
        raise ValueError('Invalid regenerated pseudo labels')
    (output/'teacher_models.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return pseudo


def run(config):
    if config['sample_weight'] != 1.0:
        raise ValueError('This baseline requires equal unit weights')
    out = project_path(config['output']); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(project_path(config['input']))
    if frame.record_id.duplicated().any() or not frame.label_source.isin(['measured', 'predicted']).all():
        raise ValueError('Duplicate IDs or unknown label provenance')
    stage1 = json.loads(project_path(config['stage1_settings']).read_text())
    numeric, categorical = config['numeric_features'], config['categorical_features']
    if not numeric and not categorical:
        raise ValueError('Empty feature list')
    if set(numeric+categorical) & {'target', 'gamma_wet', 'gamma_dry', 'wet_density', 'dry_density', 'label_source', 'source_model'}:
        raise ValueError('Target/provenance columns cannot be features')
    required = list(dict.fromkeys(['x', 'y', 'elevation', *numeric]))
    if not np.isfinite(frame[required].to_numpy(float)).all() or not np.isfinite(frame.target).all():
        raise ValueError('Missing/non-finite features or targets')
    measured = frame.loc[frame.label_source == 'measured'].copy()
    candidates = frame.loc[frame.label_source == 'predicted'].copy()
    rows = []
    for mode in config['validation_modes']:
        for seed in config['split_seeds']:
            dest = out / f'{mode}_{seed}'; dest.mkdir(exist_ok=True)
            validation = dict(mode=mode, 
                              test_size=config['test_size'], validation_size=config['validation_size'], random_state=seed)
            parts = split_data(measured, validation, target_column='target')
            train, val, test = parts
            heldout = pd.concat([val, test])
            if set(zip(train.x, train.y)) & set(zip(heldout.x, heldout.y)):
                raise ValueError('Same coordinates appear in training and heldout measurements')
            safe = exclude_heldout(candidates, heldout)
            pseudo = regenerate(train, safe, stage1['features'], stage1['config']['xgb_parameters'], config, seed, dest)
            augmented = pd.concat([train, pseudo], ignore_index=True)
            pd.concat([p[['record_id', 'boring_id', '_validation_group']].assign(split=n)
                       for n, p in zip(('train', 'validation', 'test'), parts)]).to_csv(dest/'measured_split.csv', index=False)
            augmented.to_csv(dest/'training.csv', index=False)
            audit = dict(test_labels_used_in_teacher=False, validation_labels_used_in_teacher=False,
                         reused_saved_pseudo_targets=False, same_weight=True,
                         heldout_borehole_overlap=len(set(augmented.boring_id)&set(heldout.boring_id)))
            assert audit['heldout_borehole_overlap'] == 0
            (dest/'leakage_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            for kind, learning in [('measured_only', train), ('D_first_augmented', augmented)]:
                model, pre, prediction = fit_evaluation((learning, val, test), numeric, categorical, config['xgb_parameters'])
                row = dict(mode=mode, seed=seed, model=kind, train_count=len(learning),
                           measured_train=len(train), pseudo_train=len(learning)-len(train),
                           validation_count=len(val), test_count=len(test), test_measured=len(test), test_pseudo=0,
                           best_n_estimators=model.best_iteration+1, **regression_metrics(test.target, prediction))
                rows.append(row)
                result = test[['record_id', 'boring_id', 'x', 'y', 'elevation', 'target']].copy()
                result['prediction'] = prediction
                result.to_csv(dest/f'{kind}_predictions.csv', index=False)
                if kind == 'D_first_augmented':
                    model.save_model(dest/'model.json'); joblib.dump(pre, dest/'preprocessor.joblib')
            print(f'Completed {mode} seed={seed}', flush=True)
    scores = pd.DataFrame(rows); scores.to_csv(out/'metrics.csv', index=False)
    summary = scores.groupby(['mode', 'model'])[['r2', 'rmse', 'mae']].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    selected = scores.loc[(scores['mode']==config['primary_validation']) & (scores.model=='D_first_augmented')]
    trees = int(selected.best_n_estimators.median())
    model, pre = fit_final(frame, numeric, categorical, config['xgb_parameters'], trees)
    final = out/'final'; final.mkdir(exist_ok=True)
    model.save_model(final/'model.json'); joblib.dump(pre, final/'preprocessor.joblib')
    metadata = dict(training_rows=len(frame), measured_rows=len(measured), pseudo_rows=len(candidates),
                    numeric_features=numeric, categorical_features=categorical, target='gamma_wet',
                    unit='kN/m3', sample_weight=1.0, trees=trees,
                    evaluation='See split-specific metrics; no heldout score for all-data final fit')
    (final/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(summary.to_string())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/gamma/stage2.json')
    args = parser.parse_args()
    run(json.loads(project_path(args.config).read_text()))


if __name__ == '__main__':
    main()
