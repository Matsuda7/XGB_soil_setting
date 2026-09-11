# XGB_gamma_setting

KuniJibanの土質試験XMLから湿潤・乾燥密度を抽出し、単位体積重量 gamma を目的変数とするXGBoost用データを作成するプロジェクトです。

## 実行手順

```bash
python download_kunijiban.py --max-records 1 --delay 0
python extract_strength.py
python make_distributions.py
```

通常の取得では、サーバー負荷を抑えるため逐次取得・2.5秒待機を使用します。

## 出力

```text
output/strength_data.csv
output/summary.csv
output/distributions/gamma_wet_kn_m3.csv
output/distributions/gamma_dry_kn_m3.csv
output/distributions/model_dataset_wet_gamma.csv
output/distributions/model_dataset_dry_gamma.csv
output/distributions/gamma_wet_kn_m3.png
output/distributions/gamma_dry_kn_m3.png
output/distributions/gamma_wet_kn_m3_map.png
output/distributions/gamma_dry_kn_m3_map.png
output/distributions/distribution_summary.csv
```

`model_dataset_wet_gamma.csv` と `model_dataset_dry_gamma.csv` は、先頭列 `target` を目的変数とする1行1試料の学習用CSVです。

## 単位変換

XMLの密度値を `g/cm3` として、単位体積重量を `kN/m3` へ変換します。

```text
gamma = density_g_cm3 * 9.80665
```

湿潤密度から `gamma_wet_kn_m3`、乾燥密度から `gamma_dry_kn_m3` を作成します。元XMLの単位が異なる場合は、この換算前提を見直してください。

## データ配置

```text
input_xml/strength/*.xml
metadata/boring_list.csv
logs/
output/
```

XMLや生成CSV・ログは研究データのためGit管理対象外です。

## 粘着力について

このプロジェクトでは単位体積重量に集中します。現在のXMLにある `せん断強さ_全応力` は粘着力ではないため、粘着力の変換・学習は扱いません。

## テスト

```bash
python -m unittest -v test_extract_strength.py
```
