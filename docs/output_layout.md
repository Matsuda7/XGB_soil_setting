# データと結果の保存先

入力・前処理データは `data/`、解析結果は `results/実行開始日/` に保存します。
日付は日本時間の `YYYYMMDD` で、日付をまたいでも同じ統合実行の保存先は変わりません。

```text
results/
  20260918/
    c/
      grid/                   # 分布図・二次元テキスト・格子値
      model/                  # 学習モデル・重要度・評価図
      evaluation/             # 孔単位・空間評価
      n_models/               # N値モデル（使用する場合）
      config.json
      run_summary.json
    gamma_wet/                # dry選択時はgamma_dry/
      grid/                   # 分布図・二次元テキスト・格子値
      models/wet/             # 選択した側のモデル・評価（dryならmodels/dry/）
      n_comparison/           # N値条件比較・評価
      n_models/               # N値モデル（使用する場合）
      config.json
      run_summary.json
    phi/
      grid/                   # N値・φ分布図、φ二次元テキスト、チャンク
      model/                  # モデル・評価指標・評価図・重要度
        evaluation/           # 孔単位・空間評価
      config.json
      run_summary.json
    archive/                  # 同日の再実行前の対象ディレクトリを退避
    run_summary.json           # 最後の統合コマンドの状態
```

`config/pipeline.json` の `targets` でc・gamma・phiを選びます。
`config/gamma/prediction.json` の `gamma_kind` で `wet`（湿潤）または `dry`（乾燥）を選びます。
gammaの比較・学習・分布は選択した側だけ実行します。共通前処理では両方の学習用CSVを作成します。
`--gamma-kind` を指定した場合のみconfigの選択を上書きします。

同じ日・同じ対象の再実行では、既存結果を `archive/<対象>_<時刻>_<識別子>/` へ移してから作成します。
別の対象や湿潤・乾燥の結果は保持します。失敗時も当該対象の `run_summary.json` に状態を記録します。
従来の `runs/`・`latest.json` は作成しません。基盤面作成は廃止しました。
試験地点図を作る独立コマンドの保存先は `config/test_locations.json` で指定します。

## 二次元テキスト

- `c/grid/c_effective_kpa.txt.gz`：有効粘着力（kPa）
- `gamma_wet/grid/gamma_wet.txt.gz`：湿潤単位体積重量（kN/m³）
- `gamma_dry/grid/gamma_dry.txt.gz`：乾燥単位体積重量（kN/m³）
- `phi/grid/phi_10m.txt.gz`：内部摩擦角（度）

いずれも日付ディレクトリ以下のパスです。ヘッダーなし・空白区切り・小数点以下6桁のgzip圧縮行列です。
全解像度（現在10 m）で、先頭行は南端、先頭列は西端です。行は北向き、列は東向きに増えます。
`prop=0` のセルは0です。c・gammaの `.npy` は領域外をNaNで保持します。
c・gammaのテキスト出力は各 `prediction.json` の `write_text_matrix` で切り替えます。

## 元データと生成データ

| 保存先 | 内容 |
|---|---|
| data/raw/ | 収集した元データ |
| data/interim/ | 抽出表・加工途中データ |
| data/training/ | 学習入力データ |
| data/cache/ | 再利用キャッシュ |
| logs/ | 実行ログ |

2026-09-18に過去の生成結果、中間・学習データ、キャッシュ、ログを削除しました。元データと設定は保持しています。
