# XGBoost / TabPFNの空間CV比較

Built with PriorLabs-TabPFN

## 既存解析の調査と変更方針

`run_analysis.py` は `module/pipeline.py` を通して収集・前処理・分布推定を実行する。`distributions.py` がc / φ / γへ振り分ける。従来コマンド、設定、モデル保存先は変更していない。新しい `--stage compare` は比較専用の設定と保存先を使用する。

| 対象 | 学習データと目的変数 | 既存の評価・予測 |
|---|---|---|
| c | `c_model_a.py` がCU-bar/CDの有効粘着力を抽出。SOILTESTLISTの切片または明示的c′、kPa。座標変換、重複・不正値・DEM範囲・深度で選別 | 孔単位のtrain/validation/test。`n_feature_pipeline.py` のnested cross-fitで推定Nを作る場合あり。分布は負値を0へ補正 |
| N / φ | `spt_dataset.py` が標準貫入試験を測定単位へ展開し、N>50を50へ上限処理。`phi_training.py` が深度・N=50条件を適用 | 孔単位5-fold CV。内側の孔分割でearly stoppingの木数を選択し、外側訓練全体でrefit。予測Nを[0,50]へclip。`phi_prediction.py` で `min(sqrt(15*N)+15, PHI_MAX_DEG)` に換算 |
| γ湿潤・乾燥 | `soil_tests.py` / `gamma_data.py` が密度試験を抽出。密度[g/cm³] × 9.80665 → γ[kN/m³]。湿潤・乾燥を独立に作成 | `gamma_training.py` の孔単位holdout、validationでearly stopping。推定Nや追加特徴の別実験も存在。通常分布は指定したwetまたはdry |

既存の説明変数は `config/{c,phi,gamma}/model.jsonc` で選択する。深度、測定標高、地形勾配・曲率、地質symbol、J-SHIS、土壌図SSGなど。cでは推定Nを使用し、γでは条件に応じてNを除外・付加する。`xgb_common.py` は訓練側median補完、カテゴリの最頻値補完・未知カテゴリを無視するone-hot、回帰指標を提供する。`gamma_features.model_frame` はカテゴリコードの型を統一する。NのSolphyJ特徴には専用の欠損値処理もある。

XGBoostは設定済みハイパーパラメータとearly stoppingを使用し、標準経路に一般的なハイパーパラメータ探索はない。既存のR²/RMSE/MAE、JSONモデル、joblib前処理、評価CSV、特徴重要度、散布図、地図・行列・圧縮CSVを保存する。通常経路の残差符号は一部で実測−予測であり、新しい比較経路は明確に予測−実測とする。

**調査時点の標準経路に10 km空間CVはない。** 孔分割でも近隣孔の空間相関は残る。さらに既存N特徴生成の除外は孔・近接座標ベースで、空間ブロック全体の除外ではない。この問題を作業中に報告し、既存経路を変えず比較専用に空間CVを追加した。

## 比較プロトコル

- 設定は `config/comparison.json`。4対象共通の既定説明変数はx/y、depth、sample_elevation、slope、curvature、jshis_avs30、jshis_arv、symbol、jshis_jcode。
- これは**地理・深度特徴での比較条件**であり、既存cの推定Nありモデルや既存γの特徴構成と同一ではない。両比較モデルの間では完全に同一。
- 既存の観測学習CSVを再利用。既定深度0–10 m、N=50を含む。除外行と入力ハッシュ、学習スナップショットを保存。合成データで試すときは `data_kind="synthetic"` を設定し、実測評価と区別する。疑似ラベル、推定N、土質試験から目的値を推測できる特徴を受け付けない。
- EPSG:6675、原点(0,0)、`floor(x/10000), floor(y/10000)` の10 kmブロック。seed=42の5-fold。孔ID・source孔ID・比較孔IDが複数ブロックにまたがればブロックを結合。同一ブロック・同一孔を分断しない。独立グループ不足時のランダムCVへのfallbackはない。
- 分割を一度だけ作成し、両モデルで共有。各外側訓練データでfitした**同じ**前処理結果（dense float32）を両モデルへ入力。検証の正解値は評価時だけ使用。
- XGBoostは事前固定パラメータ（500 trees等）、TabPFNは8 estimators。比較中の探索・early stoppingなし。外側検証からパラメータを選ばない。これは最適化済みモデル同士の優劣や一般的な性能保証ではない。最適化する場合は別の内側空間CVが必要。
- 全foldのOOFを連結した指標とfold平均を分離。`bias=mean(predicted-actual)`。評価は補正前予測。失敗したモデルの不完全なOOFを集計表へ載せない。
- NのOOFにはφ換算値を別列保存するが、φの評価指標は計算しない。土質試験CSVにφの項目があっても、NのOOFとの対応・試験条件を検証せずφ実測精度としない。
- 学習・推論の秒数、前処理時間を保存。TabPFNは推論時にも計算を行うため、fit時間だけで速度比較しない。初回ロードもfit時間に含む。両モデルの計算予算は同一ではない。

## TabPFNのバージョン・導入

公式 [TabPFN 2.2.1配布物](https://pypi.org/project/tabpfn/2.2.1/) と [v2回帰重み](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) を採用。最新のモデルを暗黙選択しない。確認環境はPython 3.12.7、torch 2.5.0、scikit-learn 1.5.2、XGBoost 3.0.5。2.2.1の配布メタデータはtorch>=2.1,<3、sklearn>=1.2,<1.7。既存のrequirements.txtは変更しない。

```bash
python -m venv --system-site-packages .venv-tabpfn
.venv-tabpfn/bin/python -m pip install -r requirements-tabpfn.txt
mkdir -p data/models/tabpfn
curl -fL 'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt' \
  -o data/models/tabpfn/tabpfn-v2-regressor.ckpt
```

新しい仮想環境へ導入する。今回の実証は `/tmp/soil-tabpfn-venv` で行い、元のPython環境には追加していない。重みは今回認証なしで取得できた。実行時はローカル重みを必須とし、勝手にダウンロードやクラウド推論はしない。重みのSHA-256を記録する。保存した `.tabpfn_fit` は重み本体を含まないため、再読み込みにも同じ重みが必要。

v2はPrior Labs License（追加帰属条件あり）。配布・論文掲載時は公式モデルカードと同梱ライセンスに従う。新しいモデル系列は利用条件や認証方法が異なるため、[公式モデル一覧](https://docs.priorlabs.ai/models)と[重みの取得手順](https://docs.priorlabs.ai/models/accessing-model-weights)を別途確認する。

GPUが利用可能なら自動でCUDA、なければCPU。v2のCPUは訓練1000行、CUDAは10000行、one-hot後500特徴を上限として事前検査する。超過時は失敗理由を保存し、データを削減しない。今回のNは17,248行なので、CPUのみならずv2の学習上限にも抵触する。**NのTabPFN実評価には大規模対応の新しいモデル系列とGPUによる別途検証が必要。この実装はv2のみ対応する。**

`low_memory`、`memory_saving_mode=True`、バッチ推論を使用。OOM時はバッチを小さくするかメモリを増やす。バッチ分割は学習行数の上限を解決しない。GPUはこの環境では使用不可で、CUDA実機試験は未実施。

## 実行方法

従来の `python run_analysis.py` と従来のstageはそのまま使用可能。

```bash
# 元の環境、TabPFN未導入でも動く
python run_analysis.py --stage compare --models xgboost

# TabPFN単独（実データNはv2上限のため除外を明示）
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-tabpfn/bin/python run_analysis.py \
  --stage compare --models tabpfn --comparison-targets c gamma_wet gamma_dry

# 両モデル・全対象（NのTabPFNは理由を保存し、終了コード2）
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-tabpfn/bin/python run_analysis.py --stage compare

# 両モデル・実行可能な3対象だけ
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-tabpfn/bin/python run_analysis.py \
  --stage compare --comparison-targets c gamma_wet gamma_dry

# 別設定ファイル、または設定確認
python run_analysis.py --stage compare --comparison-config config/comparison.json --plan
```

`models` を設定ファイルで選ぶことも可能。`--models` は上書き。`--comparison-targets` は比較専用で、従来の `--targets` と混用しない。

## 出力

`results/YYYYMMDD/comparison/HHMMSS_<unique>/` を新規作成し、既存結果を上書きしない。

- ルート: `config.json`, `environment.json`, `environment_freeze.txt`, `run_summary.json`, `metrics_summary.csv`, `metrics_by_fold.csv`, `oof_predictions.csv`
- 各対象: `dataset.json`, `training_snapshot.csv.gz`, `excluded_rows.csv`, `cv_splits.csv`, 指標CSV、散布図・残差図・fold別図、`status.json`
- 各対象/モデル/fold: モデル、前処理、実効パラメータ、予測CSV。失敗時は`failure.json`
- 実行環境: Git SHAとdirty状態、ソースハッシュ、パッケージ版、CPU/GPU、スレッド条件、重みハッシュ。Git SHAだけでは未コミット変更を表せないため状態も参照する。

終了コード0は全要求処理成功、2は一部未完了。実行可能モデルの結果は残すが、不完全なモデルと比較して勝者を決めない。

## 分布推定

CV完了後だけ、設定の `distribution.enabled=true` で全観測データに再fitし、両モデルの分布を作成する。失敗がある全対象実行では分布へ進まない。Nを含む今回の全対象設定では分布は実行できないため、実行可能対象を明示する。

既定は `config/spatial.json` の共通grid/DEM/maskを使用し、`gamma_features.enrich` で地理特徴を付加。10 mメッシュ、depth_m（既定5 m）と説明変数を共有する。各対象/モデルの`grid/*.csv.gz` にx,y,depth,`predicted_raw`、Nではφ換算値を追加。`difference/*.csv.gz` はTabPFN−XGBoost。地図PNGは100 m表示間隔で、CSVは全予測点を保持する。

任意の `distribution.feature_csv` で事前作成した共通グリッド特徴CSVも渡せる。座標、深度、メッシュ整合、重複を検査する。入力CSVの特徴は学習時と同じ出所・単位で作成すること。共通グリッドは一度生成して両モデルへ渡す。目的変数列はモデルへ渡さない。

c/γ/Nのraw値をclipしない。負値とN>50の件数を記録する。φ換算のみ既存のNクリップ・φ上限を適用し、raw Nを別列に保持。γ乾燥>湿潤などの追加物理整合補正は行わない。全域のTabPFN推論はCPUでは高コストなので、本番実行前に小規模グリッドで時間・メモリを確認する。

## 検証と残課題

追加テストは `python -m unittest discover -s tests -p test_model_comparison.py`。共通分割、孔/ブロック保護、指標とBias符号、訓練側だけの補完、同一入力、OOF完全性、未導入時、容量制限、CSV/図、分布差分、CLIを検査する。代用モデルを使うテストはオーケストレーションの検証でありTabPFN性能実証ではない。別途公式TabPFNのCPU fit/predict、保存/再読込、合成データでCV→最終fit→分布、実データc/γのCVを実行する。

既存テスト78件には変更前HEADでも再現する3件の失敗がある（JSONCをjson.loadsで読むテスト、γ/φの現設定と期待する特徴構成の不一致、深度設定の参照先に関する旧テスト）。今回の変更と混同しない。既存の全域分布の再計算は未実施。

論文向けの次段階は、大規模対応TabPFNによるN評価、推定N特徴を使う際の内側・外側空間ブロック除外、両モデルのnested tuning、複数seed/ブロック原点・サイズの感度分析、孔/ブロック単位の不確実性評価、φ実測値の独立対応評価。現在の地理特徴・固定パラメータの1回のCVだけで一般的優位性を結論しない。

作業開始時のローカルHEADは `f29aa4be3819ad31f55ed69a7c3f56f206447074`。この作業ディレクトリのoriginは `Matsuda7/XGB_gamma_setting` で、ご指定の `Matsuda7/XGB_soil_setting` と異なる。remoteの変更・コミット・pushは行っていない。記録先は指示されたXGB_soil_settingのIssue #1。
