"""Render comparison figures and a Japanese report."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
data = pd.read_csv(out / 'comparison.csv')
table = data.loc[data.validation == 'borehole'].set_index('dataset')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
table[['retained_measured_count', 'candidate_count']].rename(columns={
    'retained_measured_count': 'Measured', 'candidate_count': 'Predicted'}).plot.bar(stacked=True, ax=axes[0])
axes[0].set(ylabel='Wet gamma records', xlabel='')
table[['r2', 'position_only_r2', 'augmented_position_r2']].rename(columns={
    'r2': 'Lab + position teacher', 'position_only_r2': 'Position before augmentation',
    'augmented_position_r2': 'Position after augmentation'}).plot.bar(ax=axes[1])
axes[1].axhline(0, color='black', linewidth=.7)
axes[1].set(ylabel='Held-out borehole R²', xlabel='')
fig.tight_layout()
fig.savefig(out / 'comparison.png', dpi=170)
plt.close(fig)
labels = {'grain_size': '粒度分布', 'water_content': '含水比', 'saturation': '飽和度', 'void_ratio': '間隙比'}
lines = ['# 湿潤γの教師データ追加実験', '',
         '各物性に x・y・elevation を加え、実測の湿潤単位体積重量を予測しました。',
         'γの単位はkN/m³。elevationは既存処理と同じDEM地表標高−試料深度です。DEM範囲内、説明変数の欠損なしを対象としています。', '',
         '|条件|物性モデルの実測件数|保持する実測件数|追加予測件数|合計|物性モデルR²|位置モデル追加前R²|位置モデル追加後R²|',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for name, r in table.iterrows():
    lines.append(f'|{labels[name]}|{int(r.measured_wet_count)}|{int(r.retained_measured_count)}|{int(r.candidate_count)}|{int(r.augmented_total_count)}|{r.r2:.4f}|{r.position_only_r2:.4f}|{r.augmented_position_r2:.4f}|')
lines += ['', '## 評価方法と解釈', '',
          '- 上表は地点単位の固定ホールドアウト評価です。学習・検証・テストの内訳はcomparison.csv、試料IDは各split.csvに記録しています。乱数seedは42です。',
          '- 物性モデルは各物性＋x・y・elevation、位置モデルはx・y・elevationのみです。追加前後は同じ実測テスト試料で評価しています。物性モデルのR²は、追加後の分布モデルのR²とは異なります。',
          '- 追加後の評価では、教師モデルを学習部分だけで学習し、検証部分で木の本数を決めています。検証・テスト地点の追加候補を除外し、空間評価では該当ブロックの候補も除外しています。評価対象は予測ラベルではなく実測γです。',
          '- 各条件の欠損状況が異なるため、条件間では評価試料が完全には一致しません。1回の分割の結果であり、優劣を確定するものではありません。',
          '- 最終CSVは全実測280件を保持し、各物性モデルを実測データ全体で再学習して得た予測を追加しています。label_sourceでmeasured/predictedを区別できます。評価実験で加えた件数はevaluation_added_countで、最終追加件数とは異なります。',
          '- 粒度分布は礫分・砂分・シルト分・粘土分の4項目すべてを使用しています。各条件の追加候補は重複し得るため、条件別の件数を足し合わせないでください。',
          '- 飽和度と間隙比では追加対象が0件です。今回、粒度分布・含水比の予測ラベル追加は地点評価の位置モデル精度を改善しませんでした。既存の本番学習データには反映していません。', '',
          '## 空間ブロック評価（10 km）', '', '|条件|物性モデルR²|位置モデル追加前R²|位置モデル追加後R²|', '|---|---:|---:|---:|']
for _, r in data.loc[data.validation == 'spatial'].iterrows():
    lines.append(f'|{labels[r.dataset]}|{r.r2:.4f}|{r.position_only_r2:.4f}|{r.augmented_position_r2:.4f}|')
lines += ['', '## 再実行', '', 'プロジェクト直下で順に実行します。', '', '```bash',
          'python gamma_test/run_comparison.py', 'python gamma_test/create_augmented.py',
          'python gamma_test/report.py', '```', '',
          'comparison.csvが数値一覧、comparison.pngが比較図、各*_augmented.csvが追加後データです。',
          'baseline_*は別途実行した全実測280件・位置のみの参考結果です。']
(out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
