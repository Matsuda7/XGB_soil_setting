# 実測のみの湿潤γとφ共通の説明変数

2026-09-26：通常のγパイプラインを、実測湿潤密度から換算したγのみで学習する設定に変更しました。疑似γを使用する過去の二段階実験とは別です。

- `config/gamma/prediction.json`: `gamma_kind=wet`、`distribution_model=no_n`
- `config/gamma/n_comparison.json`: `enabled=false`。予測N値の追加や実測N値の対応有無による絞り込みを行いません。
- `config/gamma/model.jsonc`: 変更時点のφと同じ説明変数を指定。

数値変数：depth、jshis_avs30、jshis_arv、slope、curvature、sample_elevation。
カテゴリ変数：symbol、jshis_jcode、soil_inventory_ssg。
`sample_elevation` は各推定モデルで共通の試料標高（DEM地表標高−深度）であり、N値を使って計算する変数ではありません。
x・yは座標付与や空間分割には使いますが、現在は説明変数には入れません。代表粒度・含水比・予測N値は説明変数に含めません。
土壌分類の空間結合はφと同じ処理を使い、未取得・競合はUNKNOWNとします。カテゴリのone-hot化と数値の中央値補完は学習部分で適合します。

通常の実行は `python run_analysis.py --targets gamma --gamma-kind wet` です。変更前の保存モデル・分布は自動的には更新されず、再学習が必要です。

## 地点図

`python -m module.gamma_wet_locations` で元XMLから図を再生成できます。
保存先：`results/test_locations/gamma_wet_availability/`。

- オレンジ：正の有限値の湿潤密度が記録された試料の座標
- 水色：湿潤密度が欠損し、同じ試料に含水比（0以上）と礫・砂・シルト・粘土の4百分率（0〜100）がそろう座標
- 同じ座標の複数試料を集約。両条件が存在する場合は水色の外周とオレンジの中心で表示

この図はデータの保有状況を示すもので、モデル学習時の深度・DEM範囲・密度整合性の除外前のデータも含みます。したがって、オレンジの試料件数と学習件数は一致するとは限りません。水色は追加候補の表示のみで、今回の学習には使用しません。
PNG・PDF、試料別CSV、座標別CSV、件数と判定条件のsummary.jsonを保存します。
