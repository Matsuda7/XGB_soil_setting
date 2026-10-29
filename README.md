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
        └─ results/YYYYMMDD/  対象ごとの結果と run_summary.json
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
  pipeline.jsonc               # 既存データの取込元・取込先
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
  terrain.py                  # DEMによる地形量
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
  YYYYMMDD/                   # 実行開始日（日本時間）
    c/                        # 有効粘着力のモデル・評価・分布
    gamma_wet/                # 湿潤γを選択した場合
    gamma_dry/                # 乾燥γを選択した場合
    phi/                      # N値モデル・評価、φ分布
    archive/                  # 同日に再実行した対象の過去結果
    run_summary.json          # 統合コマンドの状態
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

出力対象は `config/pipeline.jsonc` の `targets` で指定します。現在の設定はc（モデルA）・gamma・phiです。
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

`config/pipeline.jsonc` の `sources` は初期取り込み完了後に空にしています。
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
取得一覧は `data/boring/boring_list.csv` に保存します。互換用の `data/input_xml` リンクは `data/raw/soiltest` を参照します（XML実体は移動しません）。

KuniJibanのXMLを追加取得する場合:

```bash
python run_analysis.py --stage collect --download
```

通常の収集は既存データを再利用します。DEM・地質図・J-SHIS・ボーリングCSVは既存のローカル資産を取り込みます。
出所や変換仕様が確定していないデータの自動ダウンロードは実装していません。
予測格子 `output4.txt` と領域マスク `prop.txt` は今回の確認時に見つからず、配置待ちです。
`data/raw/grid/` に配置するか、`config/pipeline.jsonc` に取込元を登録してください。
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
直下の `output/` は廃止しました。過去の生成結果は削除済みです。新しい結果は `results/YYYYMMDD/` に保存します。

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
現在の `config/pipeline.jsonc` の targets は `["c"]` です。γのみなら `["gamma"]` に変更します。
φも作成する場合は `["phi", "gamma"]` に変更します。cはCŪ・CDを使うモデルAを実装済みです（未実行）。

説明変数は `config/gamma/model.jsonc` の x, y, depth, surface_z, slope,
curvature, jshis_avs30, symbol, jshis_jcode。N値は使用しません。
`config/model.json` の prediction_depth_mを変更できます。
湿潤・乾燥密度[g/cm³]を9.80665倍して単位体積重量[kN/m³]に変換し、`gamma_kind` で選んだ側だけ学習します。
密度のみのXML試験も収集し、欠損・不正値・DEM範囲外の除外記録を残します。
学習表は data/training/gamma/、モデルと結果は results/YYYYMMDD/gamma_wet/ または gamma_dry/ に保存します。
選択した側の grid/gamma_wet.png または gamma_dry.png に、propの0/非0境界を重ねた分布図を出力します。
数値は10m格子のnpyと圧縮CSV、図の表示間隔は初期値100mです。
各実行は新規ディレクトリに保存し、既存結果を上書きしません。
処理状態は対象ディレクトリ内の run_summary.json に保存します。
乾燥γが湿潤γを上回る予測は補正せず、件数をprediction_summary.jsonに記録します。

## N値・γのvalidation切り替え

N値モデルは `config/phi/validation.json`、γは `config/gamma/prediction.json` の
validation を編集します。

```json
{"mode": "borehole", "test_size": 0.2,
 "validation_size": 0.2, "random_state": 42}
```

- mode=borehole：孔単位の分割。初期設定です。
- mode=row：行単位。同じ孔が学習・評価に混在し得るため未知孔への評価とは区別します。

validation_sizeはtestを除いた残りに対する割合です。分割可能なグループが不足すれば停止します。
主評価のvalidationで木数を選び、その木数で全データに再学習します。
空間検証は削除済みです。以前の評価ファイルは履歴として保持します。


## configの配置

```text
config/
  pipeline.jsonc          # 出力対象・収集設定
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

## γの湿潤・乾燥の選択

`config/gamma/prediction.json` の `gamma_kind` を `"wet"`（湿潤、初期設定）または
`"dry"`（乾燥）に設定します。N値比較・評価・モデル学習・分布作成は選択した側だけ実行します。
共通の前処理では、再利用できるよう湿潤・乾燥両方の学習用CSVを作成します。
実行時に上書きする場合は次のように指定します。

```bash
python run_analysis.py --targets gamma --gamma-kind wet
python run_analysis.py --targets gamma --gamma-kind dry
```

c・φも含めて最初から実行する場合は `python run_analysis.py --gamma-kind wet` を使います。
前処理済みなら `python 03_create_distributions.py --targets gamma --gamma-kind dry` で分布作成だけを実行できます。
選択した種類は各実行の `config.json` と `run_summary.json` に記録されます。


保存例とファイルの役割は [出力構成](docs/output_layout.md) を参照してください。
基盤面の補間・図・テキスト・元点データの作成は廃止しました。

## 深度を指定する二段階の湿潤γモデル

深度を加えた第一段階B・C・Dの疑似γを用い、第二段階のF4_AVS_JCODEにも深度を追加する独立コマンドです。
`config/model.json` の `prediction_depth_m` に地表からの深度（m）を指定します。

```bash
python -m module.gamma_stage2_distribution --stage all
```

学習済みモデルから別の深度の分布を出す場合は、設定を変更して `--stage predict` を指定します。
説明変数・評価結果・出力先は [二段階γの深度指定分布](docs/gamma_stage2_distribution.md) を参照してください。


### 分布図の実測点

N値・γ・cの予測深度は `config/model.json` の `prediction_depth_m`（m）で共通指定します。
`observed_depth_tolerance_m`（m、初期値1.0）は、実測点を重ねる深度の許容幅です。
予測深度±許容幅の両端を含め、同一座標では予測深度に最も近い試料を選び、同距離なら平均します。
学習用に絞る前の元実測データを使用します。ただし値・深度・座標が不正な試料は除外し、cは有効粘着力、γは密度から換算した対象の湿潤／乾燥単位体積重量を使用します。
地図内の実測値を背景と同じカラーマップ・色範囲の黒縁●で表示します。色範囲外の値もCSVには元の値を保存します。
表示点は図と同じディレクトリの `*_observed_overlay.csv` に保存します。これは独立した精度検証ではありません。
既存の予測結果はその計算深度を確認してから再描画してください。深度が異なる場合は予測の再計算が必要です。

## XGBoostとTabPFNの比較

既存解析を維持したまま `python run_analysis.py --stage compare` で比較経路を実行できます。
設定は `config/comparison.json`、導入・空間CV条件・容量制限・分布推定・検証は
[TabPFN比較ガイド](docs/tabpfn_comparison.md) を参照してください。
TabPFNは任意依存で、`--models xgboost` は未導入環境でも使用できます。

10 m全域分布の長時間実行・停止/再開・完成状態の確認は
[10 m分布ガイド](docs/full_distribution_10m.md) を参照してください。

解析はコマンド実行時だけ開始します。`python run_distribution.py --help` で分布用CLIを確認でき、`--plan` は解析を行いません。停止済みTabPFNの再開には `--resume` を明示します。

## TabPFNのCPUスレッド設定とorca（Torque PBS）

CPU用の共通設定を追加しました。以下は**利用者が後で実行するコマンド例**です。
今回の変更では解析・テスト実行・環境導入・重み取得・ジョブ投入を行っていません。

`config/comparison.json` の `tabpfn` に設定します（モデル選択は従来どおり `models` / `--models`）。

```json
{
  "cpu_threads": 8,
  "device": "cpu",
  "predict_batch_size": 1000
}
```

- `cpu_threads` は1以上の整数、または `null`。0・負数・小数・文字列・真偽値はエラーです。
- `null` はPBS割当を優先し、PBS外ではCPUアフィニティとOS CPU数から決定します。
  明示値もPBS割当・アフィニティ・OS CPU数を上限に制限し、制限時は警告とログを残します。
- `PBS_NODEFILE` は実行ホストの行数を数えます。複数ノードの合計を1プロセスに割り当てません。
  `PBS_NP` もあれば小さい方を採用します。PBS割当が不明、ノードファイルが読めない、
  実行ホストが含まれない場合は安全のため停止します。付属PBSスクリプトは1ノード専用です。
- CLI起動時、NumPy/PyTorch等のimport前に `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、
  `OPENBLAS_NUM_THREADS` を設定し、モデル作成・復元時に `torch.set_num_threads` を適用します。
  Pythonから直接モジュールを呼ぶ場合はimport順序に注意してください。遅い初期化は警告します。
- `device` は `auto`（CUDA優先、利用不可ならCPU）、`cpu`、`cuda`（CUDA不可ならエラー）。
  CPUスレッド設定はGPU計算自体を高速化する設定ではありません。
- `predict_batch_size` は1以上の整数。従来の `batch_size` よりTabPFN用設定を優先します。
  全域CLIの `--batch-size` はさらに優先します。省略時の従来値は256です。

既存v2モデルをCPUで使用する場合（既存の別環境・ローカル重みが必要）：

```bash
python run_analysis.py --stage compare --models tabpfn --comparison-config config/comparison.json
```

CPU向けの新規比較には `config/comparison.orca.json` を用意しました。
この設定は明示的に3.5-Fastを選び、CPU・スレッド自動・バッチ256を使用します。
公式の [3.5-Fast API](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/regressor.py) と
[依存関係](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/pyproject.toml) に基づき、
`requirements-tabpfn-fast.txt` は `tabpfn==9.2.0`、PyTorch 2.5以上を指定しています。
Python 3.10以上の**別環境**が必要です。既存の2.2.1環境・`requirements-tabpfn.txt` は変更しません。
実際の依存関係解決・モデル実行は未検証です。

```bash
/path/to/fast-env/bin/python run_analysis.py --stage compare --models xgboost tabpfn --comparison-config config/comparison.orca.json
```

モデル重みは `data/models/tabpfn/tabpfn-v3.5-fast-20260909.safetensors` に事前配置してください。
[公式の利用条件・認証・オフライン手順](https://docs.priorlabs.ai/models/accessing-model-weights) を確認して準備します。
コードはローカルファイルがなければ停止します。保存モデルの復元時も元の重みのパスを確認します。
今回、利用条件の代理同意・認証・ダウンロードはしていません。
既存v2の比較結果・学習済み状態をFastに切り替えて再開することはできません。Fastでは新規CVを実施し、
同じモデルで分布推定してください。モデル変更による精度差とスレッド変更による速度差は別に評価します。

orcaではリポジトリ直下から、準備済みPython環境を指定します。
`#PBS -l nodes=1:ppn=8` と `walltime` はスクリプト先頭で編集してください。
`cpu_threads=null` ならPython側も割当数を使用します（アフィニティが小さければその数まで）。
メモリ資源の指定方法・上限はorcaの運用規則に合わせて追記してください。

```bash
qsub -v PYTHON_BIN=/absolute/path/to/fast-env/bin/python scripts/orca_tabpfn_cpu.pbs
```

付属スクリプトの既定動作はTabPFN単独のCVです。標準出力・標準エラーは `logs/pbs/` にジョブID・時刻別に保存し、
Python起動前のエラーはPBS標準ログに残ります。スクリプトはパッケージ導入をしません。

既に比較が完了し、共通の地理特徴量キャッシュを準備してある10 m全域分布には次を使用します。
`config/tabpfn_cpu.json` は実行資源だけの上書き用です。元のCVモデル・特徴量・単位・メッシュを変更しません。

```bash
python run_distribution.py --stage tabpfn \
  --comparison-run results/YYYYMMDD/comparison/RUN_ID \
  --output results/YYYYMMDD/distribution_10m_5m \
  --runtime-config config/tabpfn_cpu.json

qsub -v PYTHON_BIN=/absolute/path/to/env/bin/python,JOB_STAGE=tabpfn,COMPARISON_RUN=results/YYYYMMDD/comparison/RUN_ID,DISTRIBUTION_OUTPUT=results/YYYYMMDD/distribution_10m_5m scripts/orca_tabpfn_cpu.pbs
```

重みと同じバージョンのPython環境を選んでください。停止フラグがある場合は自動で再開しません。
利用者が再開を決めたときだけ直接CLIに `--resume` を付けます。
新規全域分布の準備は [10 m分布ガイド](docs/full_distribution_10m.md) を参照してください。

CPUスレッド数は同時に働くソフトウェアの実行単位数であり、物理コア数と同じとは限りません。
OSやPBSが示す数には論理CPUが含まれる場合があります。増やせば比例して速くなるとは限らず、
メモリ帯域やスレッド間の競合が律速になります。追加のプロセス並列は導入せず、TabPFNの前処理並列数も1です。
PyTorchのinter-opスレッド数やプロセス全体の全スレッド数を指定する機能ではありません。

約2,000万点は既存のチャンク読み込み・memmap・バッチ推論を継続使用します。
バッチを小さくしても学習データ・モデル・前処理・地図マスクのメモリは必要です。
CPUでは長時間かかる可能性があり、スレッド並列だけで実用時間内の完了は保証できません。
モデルの学習行数・特徴量数の制限は維持し、超過時にデータ削減や制限解除はしません。
v2のN評価が容量制限で失敗する場合、スレッドを増やしても解消しません。
Fastも公式APIのCPU容量検査に従います。N評価が成功するとは未検証のため断定しません。
スレッド・デバイス・バッチ数の変更による浮動小数点差は未検証です。
厳密な再現性が必要な実行ではこれらの条件も固定し、途中で変更しないでください。

実行時は各モデル出力内の `runtime.jsonl` と `prediction_runtime.jsonl` に
設定スレッド数・適用値・PyTorch実値・PBS割当・アフィニティ・モデル・デバイス・バッチ数・
処理点数・推論秒数・点/秒を追記します。全域の総点数は開始ログと既存の進捗JSONに残します。
既存の結果・ログを今回削除・上書きしていません。

XGBoost単独ではTabPFN/PyTorchを読み込まず、CPU環境変数も変更しません。
両モデルを同一プロセスで比較する場合はBLAS環境設定がプロセス共通になりますが、
XGBoostのパラメータ、N・c・γの推定式、CV分割、欠損値処理は変更していません。
今回の変更ファイルと未検証事項は [作業記録](docs/tabpfn_cpu_implementation.md) を参照してください。
