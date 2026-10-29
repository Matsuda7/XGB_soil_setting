# 10 m全域分布の実行（2026-10-29追記）
> 最新状態（2026-10-29）: ユーザーの指示で解析を停止。TabPFN cは30,036点のcheckpointを保存済み。これ以降は明示的なコマンドによる開始・再開だけを行う。下記の「計算継続中」は過去の実行記録であり、現在は停止している。

## コマンドから実行する

`python run_distribution.py` または `python run_distribution.py --help` はヘルプだけを表示する。

```bash
# 実行内容を確認するだけ（学習・推論・結果書き込みなし）
python run_distribution.py --stage tabpfn \
  --comparison-run results/20261028/comparison/181333_184f0adb \
  --output results/20261028/distribution_10m_5m --resume --plan

# 停止したTabPFN解析を明示的に再開（TabPFN導入環境のPythonを使用）
python run_distribution.py --stage tabpfn \
  --comparison-run results/20261028/comparison/181333_184f0adb \
  --output results/20261028/distribution_10m_5m --resume
```

停止マーカーは通常の起動では解除しない。`--resume` がある場合だけ、二重起動防止ロック取得後に解除する。`--plan --resume` ではマーカーも変更しない。

新しい比較は `python run_analysis.py --stage compare --models xgboost` またはTabPFN導入環境で `python run_analysis.py --stage compare --models xgboost tabpfn --comparison-targets c gamma_wet gamma_dry`。得られた比較runを `--comparison-run` に指定し、新しい出力先で `--stage prepare` を実行すればXGBoost全域と共有特徴を作成できる。

ユーザーが500 m/100 mへの変更を希望せず、10 m全域を指定したため、学習データも予測メッシュも削減しない。

- 出力: `results/20261028/distribution_10m_5m/`
- 共通条件: 深度5 m、EPSG:6675、10 mメッシュ、6581×6481行列、有効18,517,097点。
- CV元: `results/20261028/comparison/181333_184f0adb/`
- XGBoost: c・N・γ湿潤・γ乾燥を全学習データでfit。Nからφを換算。
- TabPFN: c・γ湿潤・γ乾燥を全学習データでfit。Nはv2の対象規模を超えるため対象外。
- 共通の地理・深度特徴を`features/*.csv.gz`へ保存し、ハッシュ検証して同じCSVから両モデルを推論する。
- 模擬的な補間、学習データの間引き、粗いメッシュの10 mへの拡大は行っていない。

## Nを評価できなかった理由

Nの学習対象は17,248行、最初の外側foldの訓練は15,924行。採用したTabPFN 2.2.1/v2は10,000行までの学習条件を前提とし、今回の実装はCPUの安全制限1,000行も維持している。これはN値という目的変数を扱えないという意味ではない。版・学習規模・計算資源の組合せによる制約である。安全制限の無効化やデータ削減で成功扱いにしていない。

## 出力の読み方

各対象の`xgboost/raw.npy`または`tabpfn/raw.npy`が10 mの補正前行列。行は南から北、列は西から東。対象領域外はNaN。**TabPFNで未処理の場所もNaN**であり、`prediction_status.json`のstatusがcompleteになるまで全域分布として使わない。

`module.distribution_overview`はXGBoostの完了状態と全マスクの有限値を検査したうえで、次を作る。

- `xgboost_overview.png`: c、N、換算φ、湿潤γ、乾燥γの一覧。
- cの`bounded.npy`: max(raw,0)。原値はraw.npyに保持。
- Nの`bounded.npy`: clip(raw,0,50)。原値はraw.npyに保持。
- φは既存のmin(sqrt(15×clip(N,0,50))+15, PHI_MAX_DEG)。φ実測精度の評価ではない。
- γは補正しない。乾燥>湿潤の地点数も監査ファイルに記録。
- `xgboost_full_grid_audit.json`: 点数、範囲、負値件数、N上限超過件数、補正内容。

全域TabPFNが完成した対象のみ、`difference.npy`（TabPFN raw − XGBoost raw）と図を作る。未完成の差分図は作らない。

## 長時間実行と再開

```bash
# XGBoostと共通特徴キャッシュの作成
python -m module.full_distribution \
  --comparison-run results/20261028/comparison/181333_184f0adb \
  --output results/20261028/distribution_10m_5m --stage prepare

# 保存済みの共通特徴CSVからXGBoostを再計算
python -m module.full_distribution \
  --comparison-run results/20261028/comparison/181333_184f0adb \
  --output results/20261028/distribution_10m_5m --stage xgboost

# CPU上のTabPFN全域計算。既存のcheckpointから自動再開する。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TABPFN_DISABLE_TELEMETRY=1 \
  /tmp/soil-tabpfn-venv/bin/python -m module.full_distribution \
  --comparison-run results/20261028/comparison/181333_184f0adb \
  --output results/20261028/distribution_10m_5m --stage tabpfn \
  --cache-mode low_memory --batch-size 256

# XGBoost全域の監査・補正表示図・一覧図
python -m module.distribution_overview results/20261028/distribution_10m_5m
```

`/tmp`の仮想環境が失われた場合は比較ガイドに従って別環境に同じ版を導入する。TabPFNはc→湿潤γ→乾燥γの順に処理する。バッチごとに行列をflushしてから進捗JSONを置き換える。中断したバッチは再計算し、保存済みバッチから再開する。ファイルロックで二重起動を拒否する。

ユーザーが停止する場合は次を実行する。次のバッチ境界で状態を保存して停止する。

```bash
touch results/20261028/distribution_10m_5m/STOP_TABPFN
```

再開時は停止を確認し、STOP_TABPFNだけを削除して上記のTabPFNコマンドを再実行する。実行中に同じ出力へ別の計算を開始しない。

## CPU速度・キャッシュ方式の検証

CPUはIntel Core i7-13700、CUDAなし。湿潤γの229件・8 estimatorsで検証したところ、low_memory方式の16点推論は約156秒、fit_with_cacheの16点推論は約9.46秒、256点は約161.59秒（4 threads）だった。ただしfit_with_cacheの16点予測はlow_memoryと最大0.2943678 kN/m³異なり、同等性を確認できなかった。この変更は採用していない。比較と同じlow_memory・memory_saving_mode=Trueを維持する。

速度ベンチマークは両方式ともmemory_saving_mode=Falseで、訓練表から取ったクエリを使った診断であり、地盤精度評価ではない。実運用は評価時と同じmemory_saving_mode=Trueである。

全域c実行の初期実測は約20–26点/秒で、速度が持続してもcだけで約8–11日を要する推計。これは完了時刻の保証ではない。γも加わるため、CPUで3対象の全域図が短時間で完成するとは報告しない。進捗・速度は各対象の`tabpfn/prediction_status.json`を参照する。

## 修正・検証

全域用の新しい推論経路で、XGBoostへ疎行列が渡る一方、CVと全件fitはdense float32であることを発見した。疎行列の省略ゼロはXGBoostで欠損扱いとなるため、両モデルともdense float32へ統一した。Nの最初の256点では表現差による最大差16.6915があった。誤った初期出力は`superseded_sparse_input/`に隔離し、使用禁止理由を記した。共有特徴CSVから全対象を再計算し、初期出力を完成した比較結果として表示しない。

`tests/test_full_distribution.py`の3件で、dense表現の保持、同時起動拒否、行列再開、停止時の位置保存と未完成PNGの非生成を確認する。

## 全域計算の確認結果（2026-10-29）

修正版XGBoostの全域処理は2026-10-29 13:45 JSTに完了。4対象すべてで18,517,097セルが有限値、領域外はNaNであることを確認した。先頭・中間・末尾の共有特徴チャンクから抽出した点について、保存モデルへdense float32を直接入力した予測値と保存行列が完全一致した。

補正前cの負値は2,334,693セル、Nの負値は1,365セル、N>50は558,348セル。表示用bounded行列ではそれぞれ0または[0,50]に補正した。γの負値・乾燥γ>湿潤γは0セル。補正前値と監査JSONを保持している。

TabPFNはCPUで計算継続中。cの全域、湿潤γ、乾燥γの完了や差分図の完成はまだ確認していない。N/φのTabPFN分布は生成していない。
