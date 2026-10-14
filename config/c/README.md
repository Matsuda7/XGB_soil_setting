モデルAの説明変数はmodel.jsonc、予測はprediction.json、評価はvalidation.json、上流N値推定はn_estimation.jsonで指定します。最新の説明はdocs/soil_predictors.mdを参照してください。


c分布の実行コマンドは`python run_analysis.py --targets c`です。cの説明変数は`model.jsonc`、上流Nモデルの説明変数は`n_model.jsonc`です。共通の学習深度範囲と格子予測深度は`config/model.json`、`config/prediction.json`で指定します。対象固有の出力間隔などは`prediction.json`に残しています。実測Nが評価分割できる孔数に満たない場合は予測Nで全c試料を評価します。詳細は`docs/c_model_a.md`を参照してください。
