# XGB_soil_distribution

c（粘着力）、φ（内部摩擦角）、γ（単位体積重量）の分布作成を、1つのGitリポジトリで管理します。
統合先は `XGB_soil_distribution` フォルダです。

## 解析の流れ

```text
python run_analysis.py
        │
        ├─ 1. collect     元データ収集・配置・不足確認
        ├─ 2. prepare     共通前処理、c・φ・γの学習用データ整理
        ├─ 3. distribute  各モデルの学習・分布作成
        └─ results/      結果と run_summary.json
```

処理は順番に実行します。c・γ共通のXML抽出は1回です。
φの既存学習・格子予測を移植済みです。γはN値を使わないモデルを実装しました（解析・テスト未実行）。cはCŪ・CDを使うモデルAを実装済みです（未実行）。
未実装・不足データ・失敗を成功扱いせず、結果JSONに記録し、終了コード2を返します。
c（モデルA）・φ・γをtargetsで選択して実行するコードを整備しています。変更後の通し実行は未確認です。

## ディレクトリ

```text
run_analysis.py                # 3段階を順に実行
01_collect_data.py             # 第1段階だけ実行
02_prepare_data.py             # 第2段階だけ実行
03_create_distributions.py     # 第3段階だけ実行
config/
  pipeline.json               # 既存データの取込元・取込先
  phi/model.jsonc             # φ側の既存特徴量設定
  phi/prediction.json         # 予測格子・深度・出力設定
module/                       # 関数・共通処理（解析の本体）
  collection.py               # 元データ取り込み
  download.py                 # KuniJiban XML取得
  strength.py                 # XMLから共通試験表へ
  preparation.py              # c・φ・γの学習データ整理
  boring.py / spatial.py      # 座標・DEM・地質・J-SHIS
  spt_dataset.py              # SPT測定値の縦持ち化
  xgb_common.py               # 孔単位分割・前処理・評価・XGBoost生成
  phi_training.py             # φ側の既存SPTモデル学習
  phi_prediction.py           # 格子予測とφ換算・描画
  terrain.py / bedrock.py     # 地形量・基盤標高RBF
  distributions.py            # 各分布作成処理の呼び出し
  paths.py / pipeline.py      # パス定義・実行制御
data/
  raw/                        # ダウンロード・収集した元データ
    soiltest/strength/        # c・γ共通XML
    boring/                   # ボーリングCSV
    dem/                      # 標高行列 z.txt
    geology/                  # 地質図
    jshis/                    # J-SHIS
    grid/                     # output4.txt・prop.txt
  cache/dem/                  # DEMのメモリマップ用 z.npy
  interim/                    # 共通の中間データ
  training/
    c/candidates.csv          # 粘着力候補（定義・単位の確認前）
    phi/model_dataset.csv     # SPTモデル学習用
    gamma/wet.csv             # 湿潤γをtargetにした表
    gamma/dry.csv             # 乾燥γをtargetにした表
results/
  c/model_A/                  # 有効粘着力のモデル・評価・分布
  phi/model/                  # モデル・評価
  phi/grid_10m/               # φ・N値の格子分布
  gamma/runs/                 # 乾燥・湿潤モデル、評価、分布図
  run_summary.json            # 直近実行の各段階・対象別の状態
logs/
tests/
```

## 使い方

Python環境へ `requirements.txt` の依存を導入します。

```bash
python -m pip install -r requirements.txt
python run_analysis.py --plan
python run_analysis.py
```

出力対象は `config/pipeline.json` の `targets` で指定します。現在の設定はc（モデルA）のみです。
既存の `sources`・`soiltest_bbox` はそのままにして、この項目だけ編集してください。

```json
"targets": ["c", "phi", "gamma"]
```

φのみなら `["phi"]`、φとγなら `["phi", "gamma"]` とします。
`python run_analysis.py` と段階別の入口は、この設定に従ってデータ整理・分布作成を行います。
第1段階の収集は引き続き `sources` に登録した共通データを取り込みます。
対象を外しても過去に出力したファイルは削除しません。

`--targets` を指定した場合は、設定ファイルより優先します。
`--plan` でも同じ優先順位で対象を確認できます。
`targets` 未記載の旧設定は3種類すべてとして扱います。
空リストや `c`・`phi`・`gamma` 以外の名前は解析開始前にエラーになります。
重複指定は1回にまとめます。c・γの分布モデルが未実装である点は変わりません。

段階・対象を指定する例：

```bash
python run_analysis.py --stage collect
python run_analysis.py --stage prepare --targets c gamma
python run_analysis.py --stage prepare --targets phi
python run_analysis.py --stage distribute --targets phi
```

`01_collect_data.py` などの入口も同じ関数を呼びます。
解析パスはプロジェクトの場所から解決するため、別の作業ディレクトリから実行できます。
`--plan` はデータ取り込みもモデル学習も行いません。

## 第1段階：収集とディスク共有

`config/pipeline.json` の `sources` は初期取り込み完了後に空にしています。
通常の収集段階では統合先 `data/raw/` の配置を確認します。追加の取込元がある場合は `sources` に登録してください。
取り込みはハードリンク方式なので、同一ファイルシステム上では大容量データの実体を複製しません。
統合先には通常のファイルとして存在し、元のフォルダ名に依存するシンボリックリンクではありません。
元のファイル名を移動・削除しても、統合先のファイルは残ります。
既存取込先と取込元の内容が異なる場合は、上書きせず停止します。
異なるファイルシステムではハードリンクできないため、自動コピーに切り替えずエラーにします。

元データは不変として扱ってください。ハードリンク先を直接書き換えると同じ実体に影響します。
XMLダウンローダーは一時ファイルから置き換える方式で、既存リンクの実体を直接上書きしません。
収集履歴 `data/raw/collection_manifest.json` には取込元・サイズ・SHA-256を記録します。
元データ取り込み後の解析は、このプロジェクト内のファイルだけを参照します。
旧 `XGB_c_setting`・`XGB_phi_setting` はユーザーの指示で削除済みです。
統合済み384ファイルは削除前後にSHA-256を検証しています。収集履歴の旧パスは来歴として保持します。
互換用の `input_xml` リンクも統合先の `data/raw/soiltest` を参照します。

KuniJibanのXMLを追加取得する場合:

```bash
python run_analysis.py --stage collect --download
```

通常の収集は既存データを再利用します。DEM・地質図・J-SHIS・ボーリングCSVは既存のローカル資産を取り込みます。
出所や変換仕様が確定していないデータの自動ダウンロードは実装していません。
予測格子 `output4.txt` と領域マスク `prop.txt` は今回の確認時に見つからず、配置待ちです。
`data/raw/grid/` に配置するか、`config/pipeline.json` に取込元を登録してください。
DEMの元ファイル `Z.txt` は統合先では `z.txt` とします。

## 第2・第3段階の現状

| 対象 | 学習データ整理 | 分布作成 |
|---|---|---|
| φ | 元のSPT測定値整理を移植 | 元のXGBoost N値予測→φ換算を移植。格子・マスクの配置が必要 |
| c | 真の粘着力項目だけを候補表へ。現在は0件 | 定義・単位・教師データ・モデルを次段階で決定 |
| γ | 湿潤・乾燥密度から各targetを作成 | 説明変数・空間予測モデルを次段階で決定 |

γは密度をg/cm³と仮定し、`γ = 密度 × 9.80665` によりkN/m³へ変換します。
せん断強さを粘着力へ読み替える処理はしません。
φ側の既存手法（N値の上限50、孔単位分割、RBFの交差適合、φへの参考換算）は維持します。
詳しい元の解析仕様は [docs/phi_original.md](docs/phi_original.md) に保存しています。
同文書の旧パス・実行コマンドは移植前の記録です。統合後の入口とパスは本READMEが正です。

## Git管理

このフォルダの既存Gitを統合リポジトリとして使います。内側に別の `.git` は作りません。
コード・設定・説明書・テストとデータ用ディレクトリの空プレースホルダーを管理し、元データ・学習データ・モデル・図・ログは除外します。
GitHubの既存remoteは変更していません。コミット・push・GitHub側の名前変更はまだ行っていません。

## 検証

```bash
python -m unittest discover -s tests -v
python -m unittest -v test_extract_strength.py
```

旧互換入口 download_kunijiban.py・extract_strength.py・make_distributions.py・shared_data.py は削除しました。本体は module/ に集約し、テストも本体を直接参照します。
直下の `output/` は廃止しました。旧抽出表・学習表は `data/`、旧図・集計は `results/archive/legacy_output/` に移動しています。

## φ予測の再実行

`config/phi/prediction.json` の `existing_chunks` は `"archive"` に設定しています。
予測チャンクが既にある場合、`chunks_previous_*/chunks/` にフォルダを退避し、
通常の `chunks/` に新しく予測します。退避は移動なのでファイル内容を複製しませんが、
新しい予測結果を保存する分のディスク容量は必要です。
`"error"` にすると既存チャンクがある場合は従来どおり停止します。

統合コマンドは再学習するため、旧チャンクの自動再利用は行いません。
再実行は `python run_analysis.py --stage distribute --targets phi` です。
このコマンドでも再学習し、全格子を計算し直します。
既存の `--resume` は同じモデル・入力・設定での継続を利用者が確認した場合だけの機能です。
過去の実行ですでに再学習済みの場合、残った旧チャンクと現在のモデルの一致は保証できません。

## c・γの土質試験地点を可視化

```bash
python plot_test_locations.py
```

`config/test_locations.json` で元XML・出力先を指定します。
強度試験専用の抽出CSVでは欠落する密度試験も含めるため、元XMLを直接読み込みます。
`results/test_locations/c_gamma_test_locations.png` とPDFに、c関連の強度試験地点とγの密度試験地点を並べて出力します。
`test_locations.csv` に地点別の試験有無・座標、`summary.json` に集計、`excluded_records.csv` に座標欠損による除外を保存します。

輪郭は `config/phi/prediction.json` のprop配置・範囲・10m格子に従い、0と非0の境界を線にします。
負値や10以上の値も非0として扱い、元格子を間引きません。範囲外の地点は中抜きで表示します。
この線はpropの境界であり、別途取得した海岸線データではありません。
XMLの測地系コード00（日本測地系）・01（JGD2000）・02（JGD2011）を読み、同じ平面座標系に変換します。
コードの出典：[電子納品要領・土質試験結果一覧表データ 表2-4](https://www.maff.go.jp/j/nousin/seko/nouhin_youryou/attach/pdf/doboku-40.pdf)。
不明な測地系は推測せずエラーとします。

初回作成時点：c関連193地点、粘着力cそのもの0地点、湿潤・乾燥密度各184地点。
propの範囲外は全体で24地点。試料深度別の試験は同じ孔・位置に集約します。
図は予測分布ではなく試験地点図です。cはCŪ・CDを使うモデルAを実装済みです（未実行）。γは以下のN値を使わないモデルを実装しました（動作未検証）。


## N値を使わない乾燥・湿潤γモデル

コードを実装しましたが、今回の変更後の解析・学習・テストは未実行です。
`python run_analysis.py` で収集・整理・学習・分布作成を順に実行します。
現在の `config/pipeline.json` の targets は `["c"]` です。γのみなら `["gamma"]` に変更します。
φも作成する場合は `["phi", "gamma"]` に変更します。cはCŪ・CDを使うモデルAを実装済みです（未実行）。

説明変数は `config/gamma/model.jsonc` の x, y, depth, surface_z, slope,
curvature, jshis_avs30, symbol, jshis_jcode。N値は使用しません。
`config/gamma/prediction.json` の prediction_depth_m（初期値1m）を変更できます。
湿潤・乾燥密度[g/cm³]を9.80665倍して単位体積重量[kN/m³]に変換し、別々に学習します。
密度のみのXML試験も収集し、欠損・不正値・DEM範囲外の除外記録を残します。
学習表は data/training/gamma/、モデルと結果は results/gamma/runs/実行ID/ に保存します。
grid/gamma_wet.png・gamma_dry.png はpropの0/非0境界を重ねた分布図です。
数値は10m格子のnpyと圧縮CSV、図の表示間隔は初期値100mです。
各実行は新規ディレクトリに保存し、既存結果を上書きしません。
全格子の処理が完了した場合だけ latest.json を更新します。
乾燥γが湿潤γを上回る予測は補正せず、件数をprediction_summary.jsonに記録します。

## N値・γのvalidation切り替え

N値モデルは `config/phi/validation.json`、γは `config/gamma/prediction.json` の
validation を編集します。

```json
{"mode": "borehole", "additional_spatial": true,
 "block_size_m": 10000.0, "test_size": 0.2,
 "validation_size": 0.2, "random_state": 42}
```

- mode=borehole：孔単位の分割。初期設定です。
- mode=spatial：平面座標の格子ブロック単位の分割。全深度を同じ分割へ入れます。
- mode=row：行単位。同じ孔が学習・評価に混在し得るため未知孔への評価とは区別します。
- additional_spatial=false：追加の空間評価を実行しません。
- mode=row、additional_spatial=false：孔単位・空間評価を両方無効にします（行単位評価は残ります）。

空間ブロックの初期幅10kmは暫定値です。座標原点を基準に区切り、ブロックをランダムに
学習・validation・testへ分ける単回holdoutです。連続した地域の除外やバッファ付きCVではありません。
validation_sizeはtestを除いた残りに対する割合です。分割可能なグループが不足すれば停止します。
追加の空間評価は別モデルで実施し、主評価の木数選択や最終モデルを変更しません。
主評価のvalidationで木数を選び、その木数で全データに再学習します。
空間評価はspatial_metrics.json・spatial_test_predictions.csvに保存します。
無効化しても以前の評価ファイルは削除しません。現在のmetrics.jsonの設定を確認してください。


## configの配置

```text
config/
  pipeline.json          # 出力対象・収集設定
  spatial.json           # 共通の格子・DEM・prop・座標系
  test_locations.json    # c・γ共通の試験地点図
  c/                     # モデルAの予測・validation設定
  phi/
    model.jsonc
    prediction.json
    validation.json
  gamma/
    model.jsonc
    prediction.json      # γの予測・学習・validation設定
```

φとγはconfig/spatial.jsonを共有します。深度は各対象のprediction.jsonで指定します。
γのobserved_vs_predicted.pngには評価用データのR²を左上に表示します（φには既に表示あり）。
γ分布図のタイトルは英語、軸はメートルです。北を上にした図で、平面直角座標系VIIの
正式な軸名に従い横軸Y（東西）、縦軸X（南北）とします。
内部のデータ列x=東西、y=南北は従来どおりです。図・評価結果の再生成は行っていません。

## 保存先の整理

入力・中間・学習データはdata/、モデル・評価・分布図はresults/、実行ログはlogs/です。
[保存先と旧outputからの移動一覧](docs/output_layout.md)を参照してください。

## N値の導入比較

[対応ルール・比較条件・信頼性指標・実行設定](docs/n_comparison.md)を追加しました。
config/gamma/n_comparison.jsonのenabledで切り替えます。今回の変更後の解析・テストは未実行です。
比較のみの場合はconfig/gamma/prediction.jsonのcreate_distributionをfalseにしてください。

採用分布モデルをpredicted_nに変更しました。設定と学習条件は[比較・採用モデルの説明](docs/n_comparison.md)を参照してください。既存結果の更新・解析・テストは行っていません。

## c：モデルA

CŪ・CD試験から有効粘着力c′を予測する[モデルA](docs/c_model_a.md)を実装しました。
現在のtargetsはcです。解析・学習・テストは未実行です。c全体を実装待ちとしていた記述はモデルAには当てはまりません。

## モデル作成時の必須図表

寄与度の表・棒グラフと、学習目的値の度数分布図を必ず保存します。
[出力ファイルと算出方法](docs/model_reporting.md)を参照してください。今回の変更後の解析・テストは未実行です。

## 最新のc・γ説明変数

c・γとも試料標高と推定N値を含む9変数に変更しました。[最新設定と標高の定義](docs/soil_predictors.md)を参照してください。旧説明変数の記述よりこちらを優先します。
