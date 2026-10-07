"""Evaluate F4 plus AVS30 and categorical JCODE on saved Step 5 folds."""
import json
import numpy as np
import pandas as pd
from pyproj import Transformer
from module.paths import project_path
from module.gamma_stage2_integrated import attach_solphyj, fit_evaluation_missing
from module.spatial_features import meshcodes_250m, load_regional_jshis
from module.xgb_common import regression_metrics


def main():
    config = json.loads(project_path('config/gamma/stage2_integrated.json').read_text())
    source = project_path(config['output'])
    out = project_path('gamma_test/stage2_f4_jshis_borehole')
    out.mkdir(exist_ok=False)
    frame = attach_solphyj(pd.read_csv(project_path(config['input'])), config)
    for filename, columns in [('terrain_features.csv', ['slope', 'curvature']),
                              ('geology_features.csv', ['geology'])]:
        frame = frame.merge(pd.read_csv(source / filename)[['record_id', *columns]],
                            on='record_id', validate='one_to_one')
    stage1 = json.loads(project_path(config['stage1_settings']).read_text())
    transform = Transformer.from_crs(stage1['config']['spatial']['grid_crs'], 'EPSG:4326', always_xy=True)
    lon, lat = transform.transform(frame.x.to_numpy(), frame.y.to_numpy())
    frame['jshis_meshcode'] = meshcodes_250m(lon, lat)
    jshis = load_regional_jshis(set(frame.jshis_meshcode.str[:4]))
    frame = frame.merge(jshis, on='jshis_meshcode', how='left', validate='many_to_one')
    invalid = frame.jshis_jcode.isna() | frame.jshis_jcode.eq(0) | frame.jshis_avs30.fillna(0).le(0)
    frame.loc[invalid, ['jshis_jcode', 'jshis_avs30']] = np.nan
    frame[['record_id', 'label_source', 'jshis_meshcode', 'jshis_avs30', 'jshis_jcode']].to_csv(out / 'features.csv', index=False)
    measured = frame[frame.label_source.eq('measured')].set_index('record_id', drop=False)
    assert len(measured) == 280
    numeric = config['models']['F4']['numeric'] + ['jshis_avs30']
    categorical = config['models']['F4']['categorical'] + ['jshis_jcode']
    rows, audits = [], []
    reference = pd.read_csv(source / 'metrics.csv')
    for seed in config['split_seeds']:
        folder = source / f'borehole_{seed}'
        split = pd.read_csv(folder / 'measured_split.csv')
        train, val, test = [measured.loc[split.loc[split.split.eq(s), 'record_id']].copy()
                            for s in ['train', 'validation', 'test']]
        pseudo = pd.read_csv(folder / 'regenerated_pseudo.csv')
        pseudo = pseudo.merge(frame[['record_id', 'jshis_avs30', 'jshis_jcode']], on='record_id', validate='one_to_one')
        augmented = pd.concat([train, pseudo], ignore_index=True)
        heldout = pd.concat([val, test])
        assert not set(augmented.boring_id) & set(heldout.boring_id)
        audit = json.loads((folder / 'leakage_audit.json').read_text())
        assert audit['test_labels_used_in_teacher'] is False
        assert audit['validation_labels_used_in_teacher'] is False
        for condition, learning in [('measured_only', train), ('D_first_augmented', augmented)]:
            # Reproduce F4 before evaluating the added inputs.
            for name, nums, cats in [('F4', config['models']['F4']['numeric'], config['models']['F4']['categorical']),
                                     ('F4_AVS_JCODE', numeric, categorical)]:
                model, _, pred = fit_evaluation_missing((learning, val, test), nums, cats, config['xgb_parameters'])
                metrics = regression_metrics(test.target, pred)
                if name == 'F4':
                    ref = reference[(reference.seed == seed) & (reference.condition == condition) & (reference.model == 'F4')].iloc[0]
                    for key in ['r2', 'rmse', 'mae']:
                        assert np.isclose(metrics[key], ref[key], atol=1e-10, rtol=0), (seed, condition, key)
                rows.append(dict(seed=seed, condition=condition, model=name, train_count=len(learning),
                                 train_measured=len(train), train_pseudo=len(learning)-len(train),
                                 validation_count=len(val), test_count=len(test), **metrics))
                test[['record_id', 'target']].assign(prediction=pred).to_csv(out / f'{seed}_{condition}_{name}_predictions.csv', index=False)
            audits.append(dict(seed=seed, condition=condition, heldout_borehole_overlap=0,
                               unknown_test_jcode=int((~test.jshis_jcode.isin(learning.jshis_jcode)).sum())))
        print(f'Completed seed {seed}', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'metrics.csv', index=False)
    summary = scores.groupby(['condition', 'model'])[['r2', 'rmse', 'mae']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out / 'summary.csv')
    pd.DataFrame(audits).to_csv(out / 'leakage_category_audit.csv', index=False)
    config.update(numeric_features=numeric, categorical_features=categorical,
                  output=str(out), pseudo_source='Step 5 per-fold regenerated_pseudo.csv; not full-population labels')
    (out / 'settings.json').write_text(json.dumps(config, indent=2))
    (out / 'README.md').write_text('# F4 + AVS30 + JCODE\n\n'
        'Step 5と同じ280実測・10 km分割・seed 42～46・XGBoost設定。JCODEは学習データのみでone-hot化し未知カテゴリは全ゼロ。'
        'AVS30欠損はNaNのまま使用。疑似γはStep 5の各分割内で再学習して保存された値を再利用し、全件学習の疑似値は使用しない。重みは同一。'
        '各seed・各条件で元F4のR²/RMSE/MAEの再現一致を確認。最終モデル保存・全域予測なし。\n\n'
        f'J-SHIS欠損: 実測 {int(invalid[frame.label_source.eq("measured")].sum())}/280、疑似候補 {int(invalid[frame.label_source.eq("predicted")].sum())}/815。\n\n'
        + '```text\n' + summary.to_string() + '\n```' + '\n\n再実行: `python -m module.gamma_stage2_jshis`（出力先が存在する場合は停止）。\n')
    print(summary.to_string())


if __name__ == '__main__':
    main()
