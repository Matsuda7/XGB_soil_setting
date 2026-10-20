# モデルA：有効粘着力c′の空間予測

cの分布作成はφと同じ統合パイプラインから実行します。実行時にc試料の再準備、評価・最終学習、指定深度の格子予測を行います。

```bash
python run_analysis.py --targets c
```

準備済みデータを使って分布工程だけを実行する場合:

```bash
python run_analysis.py --stage distribute --targets c
```

モデル・N値上流モデルの設定は`config/c/model.jsonc`と`config/c/n_model.jsonc`、評価は`config/c/validation.json`、格子予測深度は`config/c/prediction.json`で指定します。初期予測深度は1 mです。
この呼び出しでは実行時に学習と分布予測が行われます。

## 目的値と対象試験

CŪ（CUb、B0523）・CD（B0524）の有効粘着力c′を使用します。
電子納品要領の土質試験結果一覧表では「せん断強さ（有効応力）」はc′、単位kN/m²です。
[項目定義：付属資料6、付6-2](https://www.pref.yamaguchi.lg.jp/uploaded/attachment/189665.pdf)
この標準XMLの項目は任意の応力条件下のせん断応力τではなく、強度定数の切片です。
以前の「この項目は粘着力として扱えない」「粘着力が0件」という解釈を訂正します。
旧試験地点図・旧集計結果は再生成していないため、この訂正がまだ反映されていません。

SOILTESTLISTの標準XMLであることを確認して、shear_strength_effectiveをc′に対応付けます。
明示的c_effectiveがあれば優先し、両者が矛盾する試料は除外します。
全応力の値、UU/CU、一軸圧縮強さからの換算値は混ぜません。
有効粘着力欠損・負値・不正深度・不正座標・重複・DEM範囲外を除外し、理由を保存します。
ゼロの粘着力は有効データとして残します。CŪ/CDの全111試験が学習可能とは限りません。
モデル予測が負値の場合は0へ制限し、件数を報告します。N値の上限50は粘着力へ適用しません。

## 説明変数と領域

config/c/model.jsoncで、x、y、depth、sample_elevation、slope、curvature、jshis_avs30、jshis_arv、symbol、jshis_jcode、n_valueを指定します。
sample_elevationは学習時にDEM標高−試料中央深度、予測時にDEM標高−指定深度です。
depthは地表からの深度（m）です。地表標高surface_zはcの直接の説明変数に入れません。
n_valueは上流N値モデルの推定値です。cの評価領域にあるN測定孔は、評価孔・地域の除外後に上流N値からも外し、交差適合して生成します。上流モデル特徴量はconfig/c/n_model.jsoncでc専用に定義します。
実測N値が同じ位置にないc試料も、予測N値を使って学習します。N値の直接対応有無でc試料を限定しません。実測N値が5孔未満しか対応しない場合は、実測Nを用いた比較評価を省略し、予測Nの条件で全c試料を評価します。実測Nが十分な孔に対応する場合のみ、同一試料による条件比較を行います。
cの評価孔・地域は上流N値モデルからも除外し、学習側N値も交差適合で生成します。
学習表のn_valueは前処理時は空欄とし、分割後に学習・評価ごとに生成した値を保存します。
詳細は[最新の説明変数](soil_predictors.md)を参照してください。
config/spatial.jsonをφ・γと共用し、同じ領域・10m格子・prop非0を予測します。
config/c/prediction.jsonで深度を指定できます（初期値1m）。画像の表示間隔は100m、数値格子は10mです。

## 評価・学習

config/c/validation.jsonで孔単位・空間評価を切り替えます。
異なるXML名でも同じ平面位置にある試料は同じグループとして扱います。
主評価のvalidationで木数を決め、全採用試料で最終モデルを再学習します。
空間評価は独立した追加評価です。主評価の木数選択には使いません。
評価図の左上にR²を表示します。model/の散布図も主評価のholdoutモデルの結果であり、最終再学習モデルの独立評価ではありません。

## 保存先

- data/training/c/model_A/model_dataset.csv：学習表、元項目・試験種別・元XMLを保持
- data/training/c/model_A/excluded_records.csv：除外記録
- data/training/c/model_A/preparation_summary.json：採用件数・孔数・目的値範囲・出典
- results/YYYYMMDD/c/evaluation/：孔単位・空間評価
- 同model/：最終モデル・前処理・重要度・評価への参照
- 同grid/：c_effective.png、c_effective_kpa.npy、圧縮CSVチャンク
- results/YYYYMMDD/c/run_summary.json：当該実行の状態

単位はkPa（kN/m²と同じ）。図は英語タイトルで、正式な平面直角座標系の軸名に従い横Y・縦X、単位m。

## 実行時のN値モデル設定

`config/c/n_estimation.json`はc専用の`config/c/n_model.jsonc`を参照し、現在のφモデル設定から独立しています。
c分布の実行で、実測Nが対応する孔数が孔単位評価に不足する場合も、全採用c試料を評価対象に残したまま、上流モデルから交差適合した予測Nを使います。実測N比較の結果は`evaluation/borehole/comparison.csv`に`unavailable`と理由を記録します。
分布作成を実行すると`results/YYYYMMDD/c/`にモデル、評価結果、格子予測を保存します。既存の同日結果はarchiveへ退避します。

このコード変更後のc学習・格子分布予測はまだ完了していません。格子予測は長時間かかるため、実行環境と時間を確保して上記コマンドを実行してください。


学習深度範囲は`config/model.json`で設定し、φ・c・γで共通に適用します。共通の予測深度は`config/model.json`、空間条件は`config/prediction.json`に置き、c固有のチャンクや表示間隔は`config/c/prediction.json`に置きます。
