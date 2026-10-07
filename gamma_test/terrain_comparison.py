"""Paired terrain-feature experiment using the previous saved sample splits."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from module.gamma_features import load_dem, terrain_at, model_frame
from module.gamma_training import fit_evaluation
from module.xgb_common import regression_metrics
from gamma_test.run_comparison import GROUPS


def main():
    base = Path(__file__).resolve().parent
    out = base / 'terrain_comparison'
    out.mkdir(exist_ok=True)
    config = json.loads((base / 'comparison_settings.json').read_text())['config']
    dem = load_dem(config)
    positions = ['x', 'y', 'elevation']
    rows = []
    for name in ('grain_size', 'water_content'):
        measured = pd.read_csv(base / f'{name}_measured.csv')
        candidates = pd.read_csv(base / f'{name}_candidates.csv')
        for frame in (measured, candidates):
            terrain = terrain_at(frame, dem, config)
            for col in ('slope', 'curvature'):
                frame[col] = terrain[col].to_numpy()
            if not np.isfinite(frame[['slope', 'curvature']]).all(axis=None):
                raise ValueError('Missing terrain values; matched comparison requires explicit exclusions')
        measured.to_csv(out / f'{name}_measured.csv', index=False)
        for mode in ('borehole', 'spatial'):
            split = pd.read_csv(base / f'{name}_{mode}_split.csv')
            indexed = measured.set_index('record_id', drop=False)
            parts = tuple(indexed.loc[split.loc[split.split == label, 'record_id']].copy()
                          for label in ('train', 'validation', 'test'))
            assert not set(parts[0].boring_id) & set(pd.concat(parts[1:]).boring_id)
            heldout = pd.concat(parts[1:])
            eligible = candidates.loc[~candidates.boring_id.isin(heldout.boring_id)].copy()
            if mode == 'spatial':
                size = config['validation']['block_size_m']
                blocked = set(map(tuple, np.floor(heldout[['x', 'y']].to_numpy()/size).astype(int)))
                blocks = np.floor(eligible[['x', 'y']].to_numpy()/size).astype(int)
                eligible = eligible.loc[[tuple(b) not in blocked for b in blocks]].copy()
            for variant, extra in [('baseline', []), ('slope_curvature', ['slope', 'curvature'])]:
                features = positions + GROUPS[name] + extra
                model, pre, prediction = fit_evaluation(parts, features, [], config['xgb_parameters'])
                pseudo = eligible.copy()
                pseudo['target'] = model.predict(pre.transform(model_frame(pseudo, features, [])))
                # Keep downstream features fixed to isolate the changed pseudo labels.
                augmented = (pd.concat([parts[0], pseudo], ignore_index=True), *parts[1:])
                _, _, downstream = fit_evaluation(augmented, positions, [], config['xgb_parameters'])
                row = dict(dataset=name, validation=mode, variant=variant,
                           measured_count=len(measured), candidate_count=len(candidates),
                           evaluation_added_count=len(pseudo), test_count=len(parts[2]),
                           augmented_position_r2=regression_metrics(parts[2].target, downstream)['r2'],
                           **regression_metrics(parts[2].target, prediction))
                rows.append(row)
                test = parts[2][['record_id', 'boring_id', 'target']].copy()
                test['teacher_prediction'] = prediction
                test['augmented_position_prediction'] = downstream
                test.to_csv(out / f'{name}_{mode}_{variant}_predictions.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'comparison.csv', index=False)
    previous = pd.read_csv(base / 'comparison.csv')
    for row in summary.loc[summary.variant == 'baseline'].itertuples():
        old = previous.loc[(previous.dataset == row.dataset) & (previous.validation == row.validation)].iloc[0]
        assert np.isclose(old.r2, row.r2, atol=1e-7)
        assert np.isclose(old.augmented_position_r2, row.augmented_position_r2, atol=1e-7)
    labels = {'grain_size': '粒度分布', 'water_content': '含水比'}
    lines = ['# 傾斜角・曲率の追加による精度変化', '',
             '既存の説明変数（各物性＋x・y・elevation）にslope・curvatureを同時に追加。',
             '前回保存した試料分割とモデル設定を使用し、追加前後で同一の実測テスト試料を評価しました。追加前のR²が前回と一致することも検証済みです。', '',
             '|条件|評価|追加前R²|追加後R²|差|予測ラベル追加後の位置モデルR²（前→後）|',
             '|---|---|---:|---:|---:|---:|']
    for (name, mode), group in summary.groupby(['dataset', 'validation'], sort=False):
        a, b = group.iloc[0], group.iloc[1]
        lines.append(f'|{labels[name]}|{mode}|{a.r2:.4f}|{b.r2:.4f}|{b.r2-a.r2:+.4f}|{a.augmented_position_r2:.4f} → {b.augmented_position_r2:.4f}|')
    lines += ['', 'boreholeは地点単位、spatialは10 kmブロック単位の分割です。乱数seedは42。単一分割での比較です。',
              '位置モデルの説明変数はx・y・elevationに固定し、物性モデルに傾斜角・曲率を追加したことによる予測ラベルの変化だけを比較しています。',
              '評価中の追加候補から検証・テスト地点を除外し、空間評価では該当ブロックも除外しています。',
              '傾斜角・曲率の欠損による件数変更はありません。粒度分布は実測272件・追加候補618件、含水比は実測276件・追加候補811件です。',
              '本番データおよび前回の追加済みCSVは変更していません。', '',
              '再実行: `python gamma_test/terrain_comparison.py`']
    (out / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out / 'settings.json').write_text(json.dumps(config, indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
