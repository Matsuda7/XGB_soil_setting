# pre_soilparameter_setting

KuniJiban由来のボーリングデータへ、GSJ「20万分の1日本シームレス地質図V2」、J-SHIS表層地盤属性、10 m DEM由来の地形量を付加し、XGBoostでSPT N値を推定する研究用パイプラインです。

10 m格子のN値はボーリング孔のN値を空間補間するのではなく、各格子点へ説明変数を与え、学習済みXGBoostモデルで直接推定します。

## 処理フロー

```text
KuniJibanボーリングデータ
  -> 01 座標検証・DEM標高/勾配/曲率付加・Point化
  -> 02 GSJ地質とJ-SHIS 250 m属性の結合
  -> 03 SPT測定値の縦持ち化
  -> 04 孔単位ランダム分割・特徴量比較・XGBoost学習
  -> 05 各10 m格子点でN値を直接予測・画像化
```

## リポジトリ構成

```text
pre_soil_parameter_setting/
├── data/
│   ├── boring/BorToCsv.csv
│   ├── geology/seamlessV2/
│   ├── jshis/
│   ├── grid/output4.txt
│   ├── dem/
│   │   ├── z.txt                 # 10 m DEM
│   │   └── z.npy                 # z.txtから初回生成されるキャッシュ
│   └── processed/
├── module/
│   ├── __init__.py
│   └── terrain.py               # DEM読込・勾配・曲率計算
├── 01_prepare_boring.py
├── 02_add_spatial_attributes.py
├── 03_clean_dataset.py
├── 04_train_xgb.py
├── 05_predict_10m_grid.py
├── prediction_config.json
└── requirements.txt
```

入力データ、生成データ、モデル、画像は`.gitignore`の対象です。GitHubではコード、設定、説明文を管理します。

## 入力データ

### 1. KuniJibanボーリングデータ

国土交通省、土木研究所、港湾空港技術研究所が共同運営する国土地盤情報検索サイト「KuniJiban」のボーリング情報です。

- 公式サイト：<https://www.kunijiban.pwri.go.jp/jp/>
- ローカルファイル：`data/boring/BorToCsv.csv`
- 座標系：平面直角座標系VII系（EPSG:6675）

1行が1本のボーリング孔を表す横持ちCSVです。

| 列 | 内容 | 単位・型 |
|---|---|---|
| `ファイル名` | ボーリング孔ID。処理後は`boring_id` | 文字列 |
| `坑口座標X` | 平面直角X座標 | m |
| `坑口座標Y` | 平面直角Y座標 | m |
| `坑口座標Z` | KuniJibanに記録された孔口標高 | m |
| `深度1`～`深度80` | 地表からSPT測定点までの深さ | m |
| `N値1`～`N値80` | SPT N値 | 無次元 |

現在のCSVは2,607孔を含み、富山県を中心とする周辺地域を対象としています。日本全国のボーリングデータではありません。ダウンロード日、抽出条件、CSV変換履歴は元CSVに含まれていないため、研究上の再現性確保には別途記録が必要です。

### 2. GSJ「20万分の1日本シームレス地質図V2」

- 提供機関：産業技術総合研究所 地質調査総合センター（GSJ）
- 配布ページ：<https://gbank.gsj.jp/seamless/use.html>
- ローカルディレクトリ：`data/geology/seamlessV2/`
- 元データの座標系：EPSG:4612
- 形式：地図区画別Shapefile

`02_add_spatial_attributes.py`はPolygon/MultiPolygonの地質ファイルを選び、ボーリングPointと空間結合します。通常は`within`を使用し、未一致の境界点だけ`intersects`で再検索します。同じスクリプト内で250 mメッシュコードを計算し、J-SHIS属性も結合します。

| 属性 | 内容 | モデルでの扱い |
|---|---|---|
| `symbol` | GSJ地質凡例記号 | カテゴリ特徴量 |
| `ser` | GSJ凡例通し番号 | `model_config.jsonc`で選択可能なカテゴリ特徴量 |
| その他の地質属性 | 年代・岩相等 | 中間データに可能な限り保持 |

`ser`の大小に連続的・物理的意味はないため、カテゴリとしてOne-Hot Encodingします。ローカルには5536、5537、5636、5637の4図郭があり、全国一括データではありません。

### 3. J-SHIS V4表層地盤データ

- 提供機関：防災科学技術研究所
- 配布ページ：<https://www.j-shis.bosai.go.jp/map/JSHIS2/download.html>
- ローカルファイル：`data/jshis/**/Z-V4-JAPAN-AMP-VS400_M250.csv`
- データ範囲：全国版`JAPAN`
- 空間単位：約250 mメッシュ

| 元列 | 処理後の列 | 内容 | モデルでの扱い |
|---|---|---|---|
| `CODE` | `jshis_meshcode` | 250 mメッシュコード | 結合・監査用 |
| `JCODE` | `jshis_jcode` | 微地形区分コード | カテゴリ特徴量 |
| `AVS` | `jshis_avs30` | 深さ30 mまでの平均S波速度 | 数値特徴量 |
| `ARV` | `jshis_arv` | Vs=400 m/s基盤から地表までの増幅率 | ケースC～Eで数値特徴量 |
| `AVS_EB` | `jshis_avs_eb` | AVS関連属性 | モデル未使用 |
| `AVS_REF` | `jshis_avs_ref` | AVS参照情報 | モデル未使用 |

全国版ファイルを保持しますが、学習データにはボーリング地点と対応するメッシュ属性だけを結合します。

### 4. 10 m DEM `z.txt`

- ローカルファイル：`./data/dem/z.txt`
- CRS：平面直角座標系VII系（EPSG:6675）
- 形式：空白区切りの二次元標高行列
- 行列形状：6,581行 × 6,481列
- 格子間隔：10 m
- 欠損値：`-200.0`

DEMの原作成機関、製品名、取得日、作成方法は現時点でリポジトリ内に記録されていません。研究成果で使用する前に、出典と加工履歴を追記してください。

| 項目 | 値 |
|---|---:|
| X最小値 | -45,800 m |
| X最大値 | 19,000 m |
| Y最小値 | 105,400 m |
| Y最大値 | 171,200 m |

配列と座標の対応は次のとおりです。

```python
x = -45800.0 + column_index * 10.0
y = 105400.0 + row_index * 10.0
```

列方向にXが増加し、行方向にYが増加します。上下反転、左右反転、転置は行いません。この配置はボーリング孔口標高との照合でも確認済みです。

初回読込時に`data/dem/z.npy`へfloat32形式でキャッシュし、2回目以降はメモリマップで参照します。

| DEM特徴量 | 内容 | 単位 | 計算方法 |
|---|---|---|---|
| `surface_z` | DEM地表面標高 | m | 最近傍10 m格子値 |
| `slope` | 地表面勾配 | degree | X・Y方向の中心差分 |
| `curvature` | 地表面曲率 | 1/m | `d²z/dx² + d²z/dy²`のラプラシアン |

対象セルまたは上下左右セルが欠損の場合、その地点のDEM特徴量は無効とします。学習時には有効なDEM近傍を持たないボーリング孔を除外します。

### 5. 10 m予測格子 `output4.txt`

`data/grid/output4.txt`は別プロジェクト`soil_parameter_setting/01_convert_spt.py`が生成した解析格子です。外部機関からダウンロードしたデータではありません。

```text
X Y legacy_surface_z legacy_bedrock_z
```

現在の`05_predict_10m_grid.py`ではX・Yを予測点座標として使用します。第3・第4列の旧RBF補間値は読み捨て、標高・勾配・曲率には`z.txt`を使用します。

### 6. `prop.txt`

`prop.txt`は予測画像の表示領域マスクです。値0を無効領域、1～9を有効領域として扱います。初期設定では次のファイルを参照します。

```text
/home/matsuda/soil_parameter_setting/data/prop.txt
```

行列形状と座標範囲は`z.txt`と一致します。環境が変わる場合は`prediction_config.json`のパスを変更します。

## 解析工程

| 工程 | スクリプト | 主な処理 | 主な出力 |
|---|---|---|---|
| 01 | `01_prepare_boring.py` | 必須列・座標検証、DEM標高・勾配・曲率付加、Point作成 | `boring_points.gpkg` |
| 02 | `02_add_spatial_attributes.py` | GSJ地質PolygonとJ-SHIS 250 m属性を連続結合 | `boring_geology_jshis.csv/.gpkg` |
| 03 | `03_clean_dataset.py` | SPT縦持ち化、測定点標高計算、N値上限制限 | `model_dataset.csv` |
| 04 | `04_train_xgb.py` | 孔単位分割、A～D比較、最終学習、評価・寄与度出力 | モデル、評価CSV・JSON・PNG |
| 05 | `05_predict_10m_grid.py` | 10 m格子への属性付加、N値直接推定 | 圧縮CSV、N値図、内部摩擦角図 |

## 学習データの作成

`01_prepare_boring.py`はボーリング座標をDEM格子へ対応させ、`surface_z`、`slope`、`curvature`をボーリングPointへ付加します。有効なDEM近傍を持たない孔はこの段階で除外します。

`03_clean_dataset.py`は`深度1/N値1`～`深度80/N値80`を1行1測定点の縦持ち形式へ変換します。

- 深度またはN値の片方だけが欠損する測定を除外
- 深度0 m以下を除外
- 負のN値を除外
- 50を超えるN値を50へ打止め
- `01`で保持したKuniJiban孔口標高`borehole_surface_z`とDEM特徴量を引き継ぐ
- `n_value_elevation = surface_z - depth`を計算

`depth`は測定点標高の計算と監査には使用しますが、XGBoostの特徴量には使用しません。`lowest_spt_elevation`と`distance_to_lowest_spt`は監査用であり、地質学的な基盤を意味せず、モデル特徴量には使用しません。

## XGBoostモデル

### 目的変数

| 列 | 内容 | 処理 |
|---|---|---|
| `n_value` | SPT N値 | 0以上を使用し、50超を50へ打止め |

### 最終モデルEの特徴量

| 列 | 内容 | 型 | 単位・扱い |
|---|---|---|---|
| `x` | 平面直角X座標 | 数値 | m |
| `y` | 平面直角Y座標 | 数値 | m |
| `surface_z` | DEM地表面標高 | 数値 | m |
| `slope` | DEM勾配 | 数値 | degree |
| `curvature` | DEMラプラシアン曲率 | 数値 | 1/m |
| `n_value_elevation` | N値測定・予測点標高 | 数値 | m |
| `jshis_avs30` | 平均S波速度AVS30 | 数値 | m/s |
| `jshis_arv` | 地盤増幅率ARV | 数値 | 無次元 |
| `symbol` | GSJ地質凡例記号 | カテゴリ | One-Hot Encoding |
| `jshis_jcode` | J-SHIS微地形区分 | カテゴリ | One-Hot Encoding |
| `ser` | GSJ凡例通し番号 | カテゴリ | One-Hot Encoding |
| `assumed_bedrock_elevation_rbf` | 仮定基盤標高のRBF補間値 | 数値 | m |

次の列は特徴量に使用しません。

```text
depth
lowest_spt_elevation
distance_to_lowest_spt
assumed_bedrock_elevation
geology_missing
jshis_missing
```

### 仮定基盤標高とRBF分布

孔ごとに、地表から最初に`N値 >= 50`となる測定点の標高を`assumed_bedrock_elevation`とします。N値50に達しない孔では、最も低いSPT測定標高`lowest_spt_elevation`を代用します。これは地質学的な基盤面の確定値ではなく、本解析上の仮定値です。

`assumed_bedrock_elevation_rbf`を選択した場合、学習孔の特徴量は孔単位5分割の交差補間で作り、評価時はテスト孔・検証孔をRBF元点から除外して、目的変数情報の漏洩を防ぎます。10 m格子分布では全学習孔を元点にします。観測孔群の外側でRBF外挿が発散しないよう、補間値は元点標高の最小～最大範囲に制限します。

- `data/processed/assumed_bedrock_points.csv`：孔別RBF元点
- `data/processed/grid_10m/assumed_bedrock_10m.txt.gz`：10 m格子RBF分布
- `data/processed/grid_10m/assumed_bedrock_10m.png`：RBF分布図

10 m予測では次式で予測点標高を作ります。`prediction_depth_m`を変更すると、同じXYでも異なる深度のN値を推定できます。

```python
n_value_elevation = surface_z - prediction_depth_m
```

### 特徴量の選択

特徴量はコメント対応の`model_config.jsonc`で直接選択します。使用する数値特徴量を`numeric_features`、カテゴリ特徴量を`categorical_features`へ記載します。データ列にない任意特徴量は`//`でコメントアウトして残し、使用時にコメントを外します。

既存の特徴量も、その行の先頭へ`//`を付ければ無効化できます。コメントによって配列末尾にカンマが残っても読み込める仕様です。

```jsonc
{
  "numeric_features": [
    "x",
    "y",
    "surface_z",
    "n_value_elevation"
    // , "assumed_bedrock_elevation_rbf"
  ],
  "categorical_features": ["symbol", "jshis_jcode"]
}
```

### 学習・評価方法

- 分割単位：ボーリング孔単位のランダム分割
- 同じ孔の測定点が学習・検証・テストへ分散しないグループ分割
- Early Stoppingで木の本数を決定
- 選択した木の本数で全測定データを最終学習
- 数値欠損：学習データの中央値で補完
- カテゴリ欠損：学習データの最頻値で補完
- カテゴリ特徴量：One-Hot Encoding
- DEM負値・範囲外フラグ：`dem_missing`として特徴量へ追加
- 寄与度：XGBoostのgainをOne-Hot後と元列単位で出力

## 10 m格子予測設定

`prediction_config.json`で予測条件を管理します。

| 設定 | 内容 | 初期値 |
|---|---|---|
| `grid_input` | 予測点座標を含む格子 | `./data/grid/output4.txt` |
| `dem_input` | 10 m DEM | `./data/dem/z.txt` |
| `dem_cache` | DEMキャッシュ | `./data/dem/z.npy` |
| `borehole_data` | 図へ重ねる最浅実測N値 | `./data/boring/BorToCsv.csv` |
| `grid_crs` | 座標系 | `EPSG:6675` |
| `prediction_depth_m` | 地表面からの予測深度 | 1.0 m |
| `grid_spacing_m` | DEM・予測格子間隔 | 10.0 m |
| `display_spacing_m` | PNG表示間隔 | 10.0 m |
| `n_plot_min` | N値図の表示最小値 | 0.0 |
| `n_plot_max` | N値図の表示最大値 | 62.0 |
| `input_chunk_size` | 1回の処理行数 | 200,000 |
| `output_chunk_dir` | 予測チャンク出力先 | `data/processed/grid_10m/chunks` |
| `property_mask` | 表示領域マスク | `prop.txt`への設定パス |

N値の学習目標と予測値は0～50です。`n_plot_max = 62`は画像カラースケールだけに適用されます。各孔の最浅実測N値を背景と同じカラーマップで重ねます。

内部摩擦角図は参考式`phi = min(sqrt(15 × N) + 15, 40)`で作成します。換算式の採用根拠と適用範囲は研究目的に応じて別途検討が必要です。

## 主な成果物

| 分類 | ファイル | 内容 |
|---|---|---|
| 前処理 | `data/processed/model_dataset.csv` | 1行1SPT測定点の学習データ |
| モデル | `data/processed/model/xgb_spt_model.json` | JSONで選択した特徴量による全データ学習モデル |
| 前処理器 | `data/processed/model/xgb_preprocessor.joblib` | 欠損補完・One-Hot変換 |
| 評価 | `data/processed/model/metrics.json` | 分割数、MAE、RMSE、R²、使用特徴量 |
| 評価 | `data/processed/model/test_predictions.csv` | テストデータの実測値と予測値 |
| 評価図 | `data/processed/model/observed_vs_predicted.png` | 実測SPT-Nと予測N値の比較 |
| 比較 | `data/processed/model/feature_set_comparison.csv/.png` | ケースA～Dの比較 |
| 寄与度 | `data/processed/model/feature_importance*.csv/.png` | gain寄与度 |
| 予測 | `data/processed/grid_10m/chunks/*.csv.gz` | X、Y、10 m格子予測N値 |
| 予測 | `data/processed/grid_10m/phi_10m.txt.gz` | ヘッダーなし二次元内部摩擦角行列（行=Y、列=X、欠損格子=0） |
| 可視化 | `data/processed/grid_10m/N_value_10m.png` | N値分布図 |
| 可視化 | `data/processed/grid_10m/phi_10m.png` | 内部摩擦角参考図 |

## データ品質と研究上の注意

- DEMの出典・製品名・取得日・加工履歴は未記録です。研究利用前に必ず明記してください。
- DEM範囲外またはDEM近傍に欠損があるボーリング孔は学習対象から除外されます。
- `n_value_elevation`はDEM標高とKuniJiban測定深度から再計算します。
- N値は50で打ち止めるため、モデルは50を超える値を表現しません。
- 孔単位ランダム分割は同一孔内の情報漏洩を防ぎますが、近接する別孔が学習側とテスト側へ分かれる可能性があります。
- 10 m間隔で出力しても、GSJ地質図、J-SHIS 250 m属性、ボーリング孔密度を超える実質的な空間精度を保証しません。

## 今後想定されるエラーと対処

- **CRS不一致**：ボーリング孔とDEMが同じ平面直角座標系であることを確認する。緯度経度を直接入力しない。
- **DEM配列の向き・範囲・形状の不一致**：X/Yの増加方向、`property_mask_bounds`、10 m間隔、行列数を入力DEMと照合する。
- **DEM端部での計算不能**：有限差分では実座標の上下左右10 mも必要なため、端から1セル以内は範囲外として扱う。
- **DEM負値・欠損セル**：補間に使う負の標高（`-200`を含む）、NaN、無限値は標高0 mへ置換し、`dem_missing=true`で元から0 m以上のセルと区別する。0と正の標高はそのまま使用する。
- **範囲外孔の扱い**：`include_outside_dem_boreholes=false`では学習から除外し、`true`では`outside_dem_fill_value`で補完して保持する。
- **古い中間成果物の混在**：DEM抽出方法または設定を変更した場合は01～04を順に再実行し、モデルも再学習する。
- **JSON設定値の型違い**：`include_outside_dem_boreholes`は真偽値、`outside_dem_fill_value`は数値で指定する。
- **浮動小数点境界**：格子境界付近の座標で範囲判定が変わり得るため、除外孔数と座標を実行ログで監査する。
- 未知地域への一般化性能を評価する場合は、別地域テストまたは空間ブロック交差検証が必要です。
- 古いモデル、評価ファイル、予測チャンクはDEM導入前の特徴量構成と互換性がありません。新しい04・05の結果に対して06を実行してください。
