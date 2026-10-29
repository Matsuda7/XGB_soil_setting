# TabPFN比較 実装・検証報告（2026-10-28）

既存XGBoost解析を維持し、独立した比較stageを追加した。詳細なコード調査・設計・導入・実行方法は [比較ガイド](tabpfn_comparison.md)。

## 実データのOOF結果

実行: `results/20261028/comparison/181333_184f0adb/`。深度0–10 m、EPSG:6675の10 km空間ブロック5-fold、seed=42、固定パラメータ、共通地理特徴、補正前予測。平均fold R²ではなくOOF全体の値。

| Target | Model | rows | R² | RMSE | MAE | Bias |
|---|---|---:|---:|---:|---:|---:|
| c | xgboost | 66 | -1.3715 | 20.0990 | 14.5477 | 8.5434 |
| c | tabpfn | 66 | -0.1888 | 14.2305 | 9.1591 | 1.9233 |
| N | xgboost | 17248 | 0.3023 | 15.7049 | 12.8512 | 1.1579 |
| gamma_wet | xgboost | 229 | -0.1915 | 1.9650 | 1.4052 | 0.1091 |
| gamma_wet | tabpfn | 229 | -0.0406 | 1.8365 | 1.2942 | 0.2454 |
| gamma_dry | xgboost | 229 | -0.1771 | 3.1037 | 2.3407 | 0.0921 |
| gamma_dry | tabpfn | 229 | -0.0380 | 2.9144 | 2.0875 | 0.4026 |
| N | tabpfn | — | 未評価 | — | — | — |

単位: cはkPa、γはkN/m³、Nは無次元。Bias=予測−実測。c/γではTabPFNの誤差が小さいが、両者ともOOF R²は負。一般的な優位性や修士論文で十分な精度が得られたとは結論しない。γのBiasはTabPFNの方が大きい。

NのTabPFNはfold 1の訓練15,924行でv2容量を超えた。全学習対象は17,248行。データ削減は行っていない。全対象実行の終了コードは2、run_summaryはincompleteであり、成功扱いにしていない。φ換算値は出力したがφ実測精度は評価していない。

## 変更ファイル

- `module/comparison_data.py`: 観測学習CSV、特徴制限、孔を保護する空間CV。
- `module/comparison_models.py`: 任意依存TabPFN、容量・device検査、バッチ推論、モデル保存。
- `module/model_comparison.py`: 共通学習・OOF・メタデータ・実行状態。
- `module/comparison_reporting.py`: 指標、散布図・残差・fold図・分布差分図。
- `module/comparison_distribution.py`: CV後refit、共有メッシュ、raw値・差分・φ換算。
- `module/pipeline.py`: compare stageとCLI選択。既存stageは維持。
- `config/comparison.json`, `requirements-tabpfn.txt`: 比較条件と任意依存。
- `tests/test_model_comparison.py`: 11件の検証。
- `.gitignore`, `README.md`, 本報告と比較ガイド: 環境除外・実行手順・制約。

## 検証結果

- 追加テスト11件成功。全89件中86件成功、2 failures + 1 error。
- 変更前HEADを `/tmp/soil-baseline-check` に展開して既存78件を実行し、同じ3件の失敗を再現。新規の回帰失敗なし。
- 既存テストの失敗: `test_c_features_include_depth`（JSONCをjson.loads）、`test_measured_config_matches_phi_without_n`（現設定との差）、`test_training_uses_same_filtered_population_and_saves_limit`（旧深度設定期待）。
- 元のTabPFN未導入環境でXGBoost単独の実データ4対象CV成功。
- 公式TabPFN 2.2.1 / v2のCPU回帰と保存・再読み込みの予測一致を確認。
- 実データの保存OOFを再検査し、両モデルのrow_id・fold・実測値の一致、孔/ブロック非混在を確認。
- 合成データの両モデルCV→全件fit→グリッド/差分CSV・PNG: `182126_7b9ed536`、status complete。data_kind=synthetic。
- 合成データのTabPFN単独CV: `182545_299310e3`、status complete。XGBoost単独CV: `182632_6e3d47be`、status complete。
- 別のTabPFN単独・分布併用試行 `182245_e31476a0` はプロセス終了143で未完了。成功例に含めない。両モデル分布の検証成功と単独CVの再試行成功は上記の別run。
- PNGファイルを読み込み検査し、cの左右散布図を目視確認。`git diff --check`・新規モジュールcompile成功。

実データ実行は追加メタデータ（ソースハッシュ・学習スナップショット等）の実装完了前に開始したため、当該runは初期版の出力項目を持つ。Git SHA/dirty状態、入力ハッシュ、設定、モデルパラメータ、fold/OOFは保存済み。最終版の追加メタデータは上記合成runで確認した。過去runを最終版で実行したとは扱わない。

## 未実行・今後の改善

- GPU実機検証: CUDA利用不可。
- NのTabPFN実評価: v2学習上限超過。大規模対応モデル系列、利用条件確認とGPUが必要。この実装はv2のみ対応。
- 全域の実データ分布生成・従来分布の全再計算: 未実施。共有グリッド処理と小規模分布は検証済み。
- 推定Nを説明変数とする比較: 未実装。ブロック全体を除外するnested生成が必要で、既存推定N特徴を流用していない。
- φ実測値との対応評価、nested tuning、複数分割での不確実性・感度分析は今後の課題。

## Gitの状態

ローカル実装のみ。コミット・push未実施。HEAD `f29aa4be3819ad31f55ed69a7c3f56f206447074`。originは `https://github.com/Matsuda7/XGB_gamma_setting.git` で、依頼先のXGB_soil_settingとは異なる。remoteは変更していない。作業記録は指定の `https://github.com/Matsuda7/XGB_soil_setting/issues/1`。

GitHub記録済み: https://github.com/Matsuda7/XGB_soil_setting/issues/1#issuecomment-6465468794 。連携ツールは403だったため、既存GitHub CLI認証で追記した。
