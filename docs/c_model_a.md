# モデルA：有効粘着力c′の空間予測

コード実装済み。解析・学習・テストは実行していません。
`python run_analysis.py --targets c` で収集確認→整理→評価・学習→分布予測を行う構成です。
現在のpipeline.jsonはtargets=["c"]です。γやφも実行する場合はtargetsを追加します。

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

config/c/model.jsoncで、x、y、sample_z、slope、curvature、jshis_avs30、symbol、jshis_jcode、n_inputを指定します。
sample_zは学習時にDEM標高−試料中央深度、予測時にDEM標高−指定深度です。
地表標高・深度・ARVはcの直接の説明変数に入れません。
n_inputは上流N値モデルの推定値です。上流モデルのみconfig/phi/model.jsoncを共用します。
実測N値が同じ位置にないc試料も、予測N値を使って学習します。N値の直接対応有無でc試料を限定しません。
cの評価孔・地域は上流N値モデルからも除外し、学習側N値も交差適合で生成します。
学習表のn_inputは前処理時は空欄とし、分割後に学習・評価ごとに生成した値を保存します。
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
- results/c/model_A/runs/<実行ID>/evaluation/：孔単位・空間評価
- 同model/：最終モデル・前処理・重要度・評価への参照
- 同grid/：c_effective.png、c_effective_kpa.npy、圧縮CSVチャンク
- results/c/model_A/latest.json：最後に完了した実行

単位はkPa（kN/m²と同じ）。図は英語タイトルで、正式な平面直角座標系の軸名に従い横Y・縦X、単位m。
