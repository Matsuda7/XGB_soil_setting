# TabPFN CPUスレッド・orca対応の変更記録

## 作業範囲

コード・設定・文書を変更した。解析、学習、推論、CV、テスト実行、パッケージ導入、
重み取得、PBS投入、計算ノード接続は行っていない。実測・解析データによる確認も行っていない。
既存結果と停止フラグには触れていない。変更はローカルのみで、commit/pushはしていない。

## 調査結果と変更理由

- 比較は `run_analysis.py --stage compare` → `module/model_comparison.py`。
  両モデルは同じ前処理結果・空間CV分割を使用する。今回はそれらの処理を変更しない。
- TabPFN v2アダプター、CPU/CUDA選択、容量検査、バッチ推論は既に存在した。
- 全域推論は `run_distribution.py` → `module/full_distribution.py`。
  キャッシュCSVのチャンク読み込み・memmap・バッチ単位の停止/再開がある。
  `torch.set_num_threads(4)` が固定されていたため共通ランタイム処理へ置換した。
- N・c・湿潤/乾燥γの目的変数・説明変数・前処理・単位・換算式は変更しない。
  既存v2のCPU容量超過はスレッド数増加では解消しない。
- PBSスクリプトは見つからなかったためTorqueの単一ノード用を追加した。
- `xgb_common` の設定読み込み関数を内容そのままで標準ライブラリ専用モジュールへ移した。
  元のimport名も維持した。これにより数値ライブラリimport前の環境変数設定が可能になる。

## 今回新規作成したファイル

| ファイル | 役割 |
| --- | --- |
| `module/tabpfn_runtime.py` | スレッド検証、PBS/アフィニティ上限、早期初期化、バッチ設定、ログ |
| `module/config_io.py` | 数値ライブラリをimportしない既存JSONCローダー |
| `config/comparison.orca.json` | 3.5-Fast・CPU・スレッド自動の新規比較設定 |
| `config/tabpfn_cpu.json` | 既存分布実行のCPU/スレッド/バッチだけの上書き設定 |
| `requirements-tabpfn-fast.txt` | 別環境用9.2.0依存指定。導入していない |
| `scripts/orca_tabpfn_cpu.pbs` | 1ノード・8CPU・24時間のTorqueスクリプト。投入していない |
| `tests/test_tabpfn_runtime.py` | 不正値、PBS、アフィニティ、PyTorch API呼出し等のテスト定義 |
| `docs/tabpfn_cpu_implementation.md` | 本記録 |

## 今回修正したファイル

| ファイル | 内容 |
| --- | --- |
| `run_analysis.py` | 元のmain入口を維持し、pipeline import前に初期化 |
| `run_distribution.py` | full_distribution import前に初期化 |
| `module/xgb_common.py` | JSONCローダーの再エクスポート。XGBoost処理は不変 |
| `module/comparison_models.py` | スレッド適用、明示的Fast選択、ローカル重み検査、推論ログ |
| `module/model_comparison.py` | TabPFN用バッチ数とfold別ログ、モデル名記録 |
| `module/comparison_distribution.py` | TabPFN用バッチ数と分布推論ログ |
| `module/full_distribution.py` | 固定4スレッド削除、復元時適用、実行設定上書き、ログ。除外モデルの表示をCV結果から生成 |
| `config/comparison.json` | v2を明記しcpu_threads=null、predict_batch_size=256を追加 |
| `tests/test_distribution_cli.py` | 新設定経路に合わせてモック設定を補完。未実行 |
| `README.md` | CPU実行・orca投入例・優先順位・依存/メモリ/速度/再現性の制約 |

着手前から未コミットだった比較・全域推論ファイルも含まれるため、Git上の新規/変更区分と
「今回新規作成」の区分は一致しない。既存の作業は削除していない。

## スレッド数と互換性

優先順位は明示cpu_threads → PBS → OS。最終値は実行ノードのPBS枠、アフィニティ、
OS CPU数の最小上限内に制限し、超過は警告する。PBSジョブ中に割当が不明なら停止する。
PyTorch intra-opスレッド数を設定する。新しいプロセス並列は導入しない。

XGBoost単独ではPyTorch/TabPFNをimportせず、BLAS環境変数も変更しない。
同一プロセスで両モデルを比較する場合はBLAS環境変数が共通になる。
XGBoost自体のパラメータ・推定ロジック・CVは変更しない。

既存はTabPFN 2.2.1/v2を維持する。新しいCPU比較設定だけが9.2.0/3.5-Fastを指定する。
モデルを黙って変更しない。Fastで新規CVを実行した結果と既存v2の再開結果を混ぜない。
モデル復元時は保存アーカイブ内の重みパスが設定と一致し、ローカルに存在することを確認する。
計算機移行で絶対パスが変わる場合の保存モデル移設は今回の対象外である。

## 構文確認と未検証事項

実施したのはPython 11ファイルの `ast.parse`、JSON 3ファイルの構文解析、
PBSスクリプトの `bash -n`、`git diff --check` のみ。いずれも成功した。
対象モジュールをimportして動かす確認やテスト実行はしていない。

以下は未検証：

- PyTorch実環境でのスレッド適用と実際の速度・メモリ。
- orcaのPBS環境変数・ホスト名表記・資源制限・ジョブ起動。
- Fast環境の依存関係解決、実際のモデル作成・保存・復元・予測。
- XGBoostを含む既存解析の実行回帰確認。
- スレッド/バッチ/デバイス変更による予測値の浮動小数点差。
- Nの容量制限解消、実際の精度、約2,000万点の完了時間。

## GitHubへの記録状況

記録先の指定は https://github.com/Matsuda7/XGB_soil_setting/issues/1 。
今回は最新の「許可する作業はコード・設定・文書等のみ」という範囲を優先し、
外部への投稿は行っていない。本記録はIssue追記用の内容も兼ねるが、GitHubには未記載である。

## 公式仕様の確認先

- https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/regressor.py
- https://github.com/PriorLabs/TabPFN/blob/v9.2.0/pyproject.toml
- https://docs.priorlabs.ai/models
- https://docs.priorlabs.ai/models/accessing-model-weights
- https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html

今回はコードの作成・修正のみを行い、学習・推論・解析・PBSジョブ投入は一切実行していません。
