# 説明変数の集計・図

説明変数に関する情報はこのディレクトリに保存します。

- `c/`: c教師データのsymbol別地点分布図（PNG/PDF）、分類別件数、地点座標、集計条件。
- `symbol/`: 教師データの有無で限定しない地質分類の全体分布図（PNG/PDF）、全分類の凡例CSV、対象範囲JSON。

## 再生成

- cの図・集計: `python -m module.explanatory_variable_summary`
- symbol全体図: `python -m module.symbol_distribution`

cの集計元は `data/training/c/model_A/model_dataset.csv`（前処理後・学習／検証分割前）。地質分類は地質図から地点に付与したsymbolで、試料そのものの土質分類とは異なります。元CSV更新後は再生成してください。

全体図は `data/raw/geology/seamlessV2` の地質ポリゴンを `config/spatial.json` の矩形範囲で切り出しています。propマスク内だけに限定した図ではありません。色は地質凡例のRGBを使用します。白地は地質ポリゴンのない領域です。
