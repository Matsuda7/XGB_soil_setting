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
現在、φの既存学習・格子予測を移植済みです。c・γの空間予測モデルは次の開発段階で実装します。
未実装・不足データ・失敗を成功扱いせず、結果JSONに記録し、終了コード2を返します。
**枠組みの統合段階であり、まだ1回の実行で3種類すべての分布が完成する状態ではありません。**

## ディレクトリ

```text
run_analysis.py                # 3段階を順に実行
01_collect_data.py             # 第1段階だけ実行
02_prepare_data.py             # 第2段階だけ実行
03_create_distributions.py     # 第3段階だけ実行
config/
  pipeline.json               # 既存データの取込元・取込先
  model_phi.jsonc             # φ側の既存特徴量設定
  prediction_phi.json         # 予測格子・深度・出力設定
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
  c/                          # 今後実装
  phi/model/                  # モデル・評価
  phi/grid_10m/               # φ・N値の格子分布
  gamma/                      # 今後実装
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

出力対象は `config/pipeline.json` の `targets` で指定します。初期値は3種類すべてです。
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

旧 `download_kunijiban.py`・`extract_strength.py`・`make_distributions.py` は互換入口として残し、本体は `module/` に集約しました。
旧 `output/` の結果は保存されていますが、統合パイプラインは新しい `data/` と `results/` を使います。
