# データと結果の保存先

直下のoutput/は廃止しました。既存ファイルを移動し、解析・図の再生成は行っていません。

| 保存先 | 内容 |
|---|---|
| data/raw/ | ダウンロード・収集した元データ |
| data/interim/ | 抽出表・加工途中のデータ |
| data/training/ | 学習に入力するデータ |
| data/cache/ | 再利用するキャッシュ |
| results/phi/model/ | 現行のN値モデル・評価・重要度 |
| results/phi/grid_10m/ | 現行のN値・φ分布図、格子値、チャンク |
| results/gamma/runs/<実行ID>/models/wet・dry/ | 実行ごとのγモデル・評価 |
| results/gamma/runs/<実行ID>/grid/ | 同じ実行のγ分布図・格子値・チャンク |
| results/gamma/latest.json | 最後に完了したγ実行のIDと状態 |
| results/test_locations/ | 試験地点図・地点一覧 |
| results/c/ | 実装待ち |
| results/archive/ | 統合前などの過去結果。現行予測には使わない |
| results/run_summary.json | 最後の統合コマンドの状態（対象はtargetsを確認） |
| logs/ | 実行ログ |

γはlatest.jsonのrunに記載されたディレクトリを参照します。新規実行が失敗しても、
latest.jsonは直前に完了した実行を指し続けます。φは従来の固定保存先を維持します。
旧outputの図は観測試験値の可視化であり、現行XGBoostの予測分布ではありません。

## 今回の移動一覧

| 元の場所 | 移動先 |
|---|---|
| output/strength_data.csv | data/interim/legacy_output/strength_data.csv |
| output/summary.csv | data/interim/legacy_output/summary.csv |
| output/distributions/cohesion.csv | results/archive/legacy_output/distributions/cohesion.csv |
| output/distributions/distribution_summary.csv | results/archive/legacy_output/distributions/distribution_summary.csv |
| output/distributions/gamma_dry_kn_m3.csv | results/archive/legacy_output/distributions/gamma_dry_kn_m3.csv |
| output/distributions/gamma_dry_kn_m3.png | results/archive/legacy_output/distributions/gamma_dry_kn_m3.png |
| output/distributions/gamma_dry_kn_m3_map.png | results/archive/legacy_output/distributions/gamma_dry_kn_m3_map.png |
| output/distributions/gamma_wet_kn_m3.csv | results/archive/legacy_output/distributions/gamma_wet_kn_m3.csv |
| output/distributions/gamma_wet_kn_m3.png | results/archive/legacy_output/distributions/gamma_wet_kn_m3.png |
| output/distributions/gamma_wet_kn_m3_map.png | results/archive/legacy_output/distributions/gamma_wet_kn_m3_map.png |
| output/distributions/model_dataset_cohesion.csv | data/training/legacy_output/model_dataset_cohesion.csv |
| output/distributions/model_dataset_dry_gamma.csv | data/training/legacy_output/model_dataset_dry_gamma.csv |
| output/distributions/model_dataset_wet_gamma.csv | data/training/legacy_output/model_dataset_wet_gamma.csv |
