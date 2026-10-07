# 解析結果

解析結果は `results/YYYYMMDD/` に保存します（日付は実行開始時の日本時間）。
その下の `c/`、`gamma_wet/` または `gamma_dry/`、`phi/` に図・二次元テキスト・モデル・評価結果をまとめます。

gammaの種類は `config/gamma/prediction.json` の `gamma_kind`（wet/dry）で指定します。
同日・同対象の再実行では既存結果を日付フォルダ内の `archive/` に退避します。
各対象と日付フォルダの `run_summary.json` で処理状態を確認できます。

詳細は [保存先の説明](../docs/output_layout.md) を参照してください。
