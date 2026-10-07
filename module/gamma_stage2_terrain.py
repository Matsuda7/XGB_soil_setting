"""Paired Step 2 terrain comparison; no final fitting or model serialization."""
import argparse
import json
import numpy as np
import pandas as pd
from module.paths import project_path
from module.gamma_features import load_dem, terrain_at
from module.gamma_stage2 import exclude_heldout, regenerate
from module.gamma_training import fit_evaluation
from module.xgb_common import regression_metrics


def run(settings, attach_features=None):
    baseline = project_path(settings['baseline_output'])
    out = project_path(settings['output']); out.mkdir(parents=True, exist_ok=True)
    config = json.loads((baseline/'config.json').read_text())
    stage1 = json.loads(project_path(config['stage1_settings']).read_text())
    frame = pd.read_csv(project_path(config['input']))
    if attach_features is not None:
        frame = attach_features(frame)
    baseline_name = settings.get('baseline_model', 'M0')
    model_names = list(settings['models'])
    category_rows = []
    terrain = terrain_at(frame, load_dem(stage1['config']), stage1['config'])
    terrain_metadata = dict(
        dem=stage1['config']['spatial']['dem_input'],
        crs=stage1['config']['spatial']['grid_crs'],
        resolution_m=stage1['config']['spatial']['grid_spacing_m'],
        slope_unit='degree', curvature_unit='1/m',
        algorithm='Bilinear DEM interpolation at center and x/y offsets of one cell; centered finite differences',
        slope='degrees(atan(sqrt(((E-W)/(2*h))**2 + ((N-S)/(2*h))**2)))',
        curvature='(E+W+N+S-4*Z)/(h*h); Laplacian, not profile/plan curvature',
        missing_policy='Existing terrain code: negative/nonfinite DEM cells -> 0; outside offset samples -> 0; outside center -> NaN. Nonfinite derived features abort this comparison, with no row removal.',
        affected_rows=int(terrain.dem_missing.sum()),
        affected_measured=int(terrain.loc[frame.label_source=='measured','dem_missing'].sum()),
        affected_pseudo=int(terrain.loc[frame.label_source=='predicted','dem_missing'].sum()),
        missing_counts=terrain[['slope','curvature']].isna().sum().to_dict())
    for col in ['slope', 'curvature']:
        frame[col] = terrain[col].to_numpy()
    if not np.isfinite(frame[['slope', 'curvature']]).all(axis=None):
        raise ValueError('Nonfinite terrain features; cannot preserve Step 1 population')
    frame[['record_id', 'x', 'y', 'slope', 'curvature']].assign(dem_missing=terrain.dem_missing).to_csv(out/'terrain_features.csv', index=False)
    measured = frame.loc[frame.label_source=='measured'].set_index('record_id', drop=False)
    candidates = frame.loc[frame.label_source=='predicted'].copy()
    reference = pd.read_csv(baseline/'metrics.csv')
    rows = []
    for seed in config['split_seeds']:
        dest = out/f'borehole_{seed}'; dest.mkdir(exist_ok=True)
        split = pd.read_csv(baseline/f'borehole_{seed}/measured_split.csv')
        parts = []
        for name in ['train', 'validation', 'test']:
            chosen = split.loc[split.split==name]
            part = measured.loc[chosen.record_id].copy()
            part['_validation_group'] = chosen._validation_group.to_numpy()
            parts.append(part)
        train, val, test = parts
        heldout = pd.concat([val, test])
        safe = exclude_heldout(candidates, heldout)
        pseudo = regenerate(train, safe, stage1['features'], stage1['config']['xgb_parameters'], config, seed, dest)
        augmented = pd.concat([train, pseudo], ignore_index=True)
        old_train = pd.read_csv(baseline/f'borehole_{seed}/training.csv')
        np.testing.assert_array_equal(augmented.record_id.to_numpy(), old_train.record_id.to_numpy())
        np.testing.assert_allclose(augmented.target, old_train.target, rtol=0, atol=1e-6)
        old_teacher = pd.read_csv(baseline/f'borehole_{seed}/teacher_split.csv')
        pd.testing.assert_frame_equal(pd.read_csv(dest/'teacher_split.csv'), old_teacher)
        assert not set(augmented.boring_id) & set(heldout.boring_id)
        teacher = pd.read_csv(dest/'teacher_split.csv')
        assert set(teacher.record_id) == set(train.record_id)
        split.to_csv(dest/'measured_split.csv', index=False)
        pseudo.to_csv(dest/'regenerated_pseudo.csv', index=False)
        (dest/'audit.json').write_text(json.dumps(dict(
            identical_step1_splits=True, identical_teacher_splits=True,
            identical_training_ids=True, regenerated_targets_match_step1=True,
            heldout_hole_overlap=0,
            teacher_uses_only_outer_train=True, sample_weight=1.0), indent=2)+'\n')
        for name, features in settings['models'].items():
            for condition, learning in [('measured_only', train), ('D_first_augmented', augmented)]:
                categorical = settings.get('categorical_features', {}).get(name, [])
                model, _, predicted = fit_evaluation((learning, val, test), features, categorical, config['xgb_parameters'])
                for column in categorical:
                    known = set(learning[column])
                    category_rows.append(dict(seed=seed, model=name, condition=condition,
                        feature=column, train_categories=len(known),
                        unseen_validation=int((~val[column].isin(known)).sum()),
                        unseen_test=int((~test[column].isin(known)).sum()),
                        unseen_test_categories=';'.join(sorted(set(test[column])-known))))
                metrics = regression_metrics(test.target, predicted)
                if name == baseline_name:
                    old = reference.loc[(reference['mode']=='borehole') & (reference.seed==seed) & (reference.model==condition)].iloc[0]
                    for metric in ['r2', 'rmse', 'mae']:
                        np.testing.assert_allclose(metrics[metric], old[metric], rtol=0, atol=1e-7)
                rows.append(dict(seed=seed, model=name, condition=condition,
                    train_measured=len(train), train_pseudo=len(learning)-len(train),
                    train_count=len(learning), validation_count=len(val), test_count=len(test),
                    best_n_estimators=model.best_iteration+1, **metrics))
                test[['record_id', 'boring_id', 'target']].assign(prediction=predicted).to_csv(dest/f'{name}_{condition}_predictions.csv', index=False)
        print(f'Completed borehole seed={seed}: all 8 conditions; {baseline_name} matches Step 1', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'metrics.csv', index=False)
    summary = scores.groupby(['condition', 'model'])[['r2', 'rmse', 'mae']].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    reference = scores.loc[scores.model==baseline_name, ['seed','condition','r2','rmse','mae']]
    deltas = scores.loc[scores.model!=baseline_name].merge(reference, on=['seed','condition'], suffixes=('', '_'+baseline_name), validate='many_to_one')
    for metric in ['r2','rmse','mae']:
        deltas[f'delta_{metric}'] = deltas[metric]-deltas[f'{metric}_{baseline_name}']
    deltas.to_csv(out/'deltas.csv', index=False)
    delta_summary = deltas.groupby(['condition','model'])[['delta_r2','delta_rmse','delta_mae']].agg(['mean','std'])
    delta_summary.to_csv(out/'delta_summary.csv')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ['r2', 'rmse', 'mae']):
        for offset, condition in [(-0.08, 'measured_only'), (0.08, 'D_first_augmented')]:
            group = summary.loc[condition].reindex(model_names)
            ax.errorbar(np.arange(4)+offset, group[(metric,'mean')],
                        yerr=group[(metric,'std')], marker='o', capsize=3,
                        label=condition)
        ax.set_xticks(range(4), model_names)
        ax.set_title(metric.upper()+' (mean +/- SD)')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=170); plt.close(fig)
    (out/'settings.json').write_text(json.dumps(dict(comparison=settings, baseline=config, terrain=terrain_metadata), indent=2)+'\n')
    if category_rows:
        pd.DataFrame(category_rows).to_csv(out/'unseen_categories.csv', index=False)
    if settings.get('custom_report', False):
        print(summary.to_string())
        return scores
    lines = ['# Step 2：第二段階の地形特徴量比較', '',
        'Step 1の孔単位分割seed 42〜46を保存済みID・順序まで再使用。第一段階はseedごとに再学習し、同じ疑似γをM0〜M3間で共用しています。第一段階の説明変数・条件は変更していません。',
        '全seedで学習ID・第一段階内部の分割・再生成疑似値がStep 1と一致し、M0の全指標も一致することを自動検証しました。検証・テスト実測を第一段階に使っていません。', '',
        '実測・疑似は同じ重み。第二段階のハイパーパラメータ・早期停止方法もStep 1と同じです。木の本数は各条件の検証実測から選択するため異なることがあります。', '',
        '|条件|モデル|R² 平均±SD|RMSE 平均±SD|MAE 平均±SD|', '|---|---|---:|---:|---:|']
    for (condition, name), row in summary.iterrows():
        values = [f"{row[(m,'mean')]:.4f} ± {row[(m,'std')]:.4f}" for m in ['r2','rmse','mae']]
        lines.append('|'+ '|'.join([condition,name,*values])+'|')
    lines += ['', 'SDは5回の標本標準偏差（ddof=1）。反復ホールドアウトのため、5分割交差検証や独立した信頼区間ではありません。RMSE・MAEはkN/m³。', '',
        '## M0との差（変更後−M0）', '', '|条件|モデル|ΔR² 平均±SD|ΔRMSE 平均±SD|ΔMAE 平均±SD|', '|---|---|---:|---:|---:|']
    for (condition, name), row in delta_summary.iterrows():
        values=[f"{row[(m,'mean')]:+.4f} ± {row[(m,'std')]:.4f}" for m in ['delta_r2','delta_rmse','delta_mae']]
        lines.append('|'+ '|'.join([condition,name,*values])+'|')
    lines += ['', 'ΔR²は正、ΔRMSE・ΔMAEは負が改善。差のSDは各seedの対応する差から計算。', '',
        '## 地形計算', '',
        '- DEM: `'+terrain_metadata['dem']+'`、CRS: '+terrain_metadata['crs']+'、格子間隔: '+str(terrain_metadata['resolution_m'])+' m。',
        '- 地点中心と東西南北1セルの位置でDEMを双線形補間。h=10 m、標高単位m。',
        '- slope = degrees(atan(sqrt(((E−W)/(2h))² + ((N−S)/(2h))²)))。単位は度。',
        '- curvature = (E+W+N+S−4Z)/h²。単位はm⁻¹。標高のラプラシアンであり、縦断・平面曲率ではありません。',
        '- 既存処理をそのまま使用：負標高（−200を含む）・非有限DEMセルは0に置換。隣接評価点がDEM外なら0、中心がDEM外ならNaN。負標高を含む0置換は端部等の傾斜・曲率に影響し得ます。',
        f"- 0置換等の影響フラグあり：実測{terrain_metadata['affected_measured']}件、疑似候補{terrain_metadata['affected_pseudo']}件。計算後の欠損：{terrain_metadata['missing_counts']}。行の除外・追加はありません。非有限値が残れば比較を停止する設計です。",
        '- 共通前処理は学習データの中央値補完ですが、今回の説明変数に欠損はなく、補完は実質的に行われません。',
        '- elevationは試料標高（既存の地表標高−深度）を保持。slope・curvatureは地表DEMから計算。', '',
        '## 保存ファイル', '',
        '`metrics.csv`：40条件実行のseed別指標・件数。`summary.csv`：平均・SD。`deltas.csv`：seed別M0差。`delta_summary.csv`：差の平均・SD。',
        '各borehole_*配下に試料分割、第一段階内部分割、再生成疑似γ、監査、テスト予測を保存。最終モデル保存・全域予測は実行していません。',
        'M0=x,y,elevation；M1=M0+slope；M2=M0+curvature；M3=M0+slope+curvature。',
        '以前の試行で使った実測データを再利用しているため、結果は説明変数選定のための比較であり、未使用の外部データによる最終検証ではありません。', '',
        '再実行：`python -m module.gamma_stage2_terrain`']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary.to_string())
    print(delta_summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/gamma/stage2_terrain.json')
    args = parser.parse_args()
    run(json.loads(project_path(args.config).read_text()))
