# 解析結果

データ入力はdata/、解析結果はresults/、実行ログはlogs/へ保存します。直下のoutput/は廃止しました。

- φ・N値の図：phi/grid_10m/。モデル・評価：phi/model/。
- γ：gamma/latest.jsonのrunが最後の完了実行IDです。
  gamma/runs/<実行ID>/grid/に図と数値、models/wet/・models/dry/にモデルと評価があります。
- 試験地点図：test_locations/。
- c：実装待ち。
- archive/legacy_output/：統合前の観測値の図・集計。現在の予測結果とは区別してください。

全体の状態はrun_summary.jsonを確認します。
詳細・移動一覧は[保存先の説明](../docs/output_layout.md)を参照してください。
