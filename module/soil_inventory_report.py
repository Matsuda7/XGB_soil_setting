"""Render Step 4-1 audit files as a Japanese report (no fitting)."""
import json
import pandas as pd
from module.paths import ROOT


def main():
    out=ROOT/'gamma_test/soil_inventory_audit'
    s=json.loads((out/'summary.json').read_text())
    attrs=json.loads((out/'soil_attributes.json').read_text())
    cv=pd.read_csv(out/'spatial_cv_categories.csv')
    counts=pd.read_csv(out/'teacher_category_counts.csv')
    sheets=pd.read_csv(out/'solphy_sheets.csv')
    lines=['# Step 4-1：土壌インベントリーとSolphyJの対応調査', '',
        '学習・物性のモデル入力・疑似γ再生成は実行していません。既存の実測280件＋疑似候補815件と、Step 1の保存済み10 km分割を使用しています。', '',
        '## 発見ファイルとデータ構造', '',
        '`data/raw/soil_inventory/石川_20.zip`内のShapeを直接読み込みました。CSVは見つからず、SolphyJは`Supplementary_Tables_SolphyJ_2025.xlsx`でした。元ファイルは変更していません。',
        f"Shape CRS={attrs['crs']}、ポリゴン数={s['map_polygons']}。属性は次のとおり。全ユニーク値はsoil_attributes.jsonに保存。", '',
        '|属性|ユニーク数|','|---|---:|']
    for c,v in attrs['unique_values'].items():lines.append(f'|{c}|{len(v)}|')
    lines += ['', 'SSerGrCDとSSG codeの完全一致で照合。SG_CD、SSbG_CDはより短い分類コードで、今回のSSG結合キーには使用しません。名称の曖昧一致・前方一致は実施していません。',
        'SolphyJは分類対応表1シート＋物性表22シート。各物性表の4行目をヘッダーとし、原尺度側のLand use、SSG code、Soil Series Group、Horizon、Number of samples、Mean、SD、Medianを読み込みました。右側のLog10統計と混同していません。',
        'Land useはPaddy / Field / Orchard / Grass / House、Horizonは1〜6。0サンプル行と統計量欠損が存在します。物性名・単位・行数・利用可能行数はsolphy_sheets.csvに記録。',
        '物性：'+', '.join(sheets.sheet)+ '。', '',
        '## コード対応', '',
        f"土壌図分類{s['map_classes']}、SolphyJコード{s['solphy_codes']}（名称{s['solphy_names']}）、直接一致{s['direct_matches']}。",
        '土壌図のみ：'+', '.join(s['unmatched_map'])+'。',
        f"SolphyJのみ：{len(s['solphy_only'])}分類。全一覧はsummary.jsonおよびcode_crosswalk.csvに保存。",
        '一致コードのうち、どの物性にも有効な平均値がない分類：'+(', '.join(s['matching_codes_without_any_mean']) or 'なし')+'。',
        'コード一致は、任意の土地利用・層位・物性に値があることを意味しません。有効値はNumber of samples>0かつMean非欠損と定義し、物性ごとの対応はproperty_code_availability.csvに記録しています。', '',
        '## 面積対応率', '']
    a=s['area']
    lines += [f"既存設定の矩形領域は{a['rectangle_km2']:.2f} km²。土壌図が覆う部分は{a['map_covered_km2']:.2f} km²。コード一致部分は{a['code_matched_km2']:.2f} km²。",
        f"矩形全域に対するコード一致率：{a['code_fraction_of_rectangle']:.2%}。土壌図で覆われた面積に限ると{a['code_fraction_of_map_covered']:.2%}。",
        f"実際の予測対象propマスク（非ゼロ格子）に対する一致率：{a['prop_mask_code_fraction']:.2%}。少なくとも1物性に有効平均がある率：{a['prop_mask_any_mean_fraction']:.2%}。",
        '矩形の面積はEPSG:6675へ投影しポリゴンunionを切り抜いて計算。propマスクは10 m格子点の等面積重みによる近似です。海域や県外を含む矩形と、実際の予測対象マスクを区別してください。', '',
        '## γ教師データとの対応', '', '|区分|総数|土壌コード取得|SolphyJコード一致|いずれかの物性平均あり|','|---|---:|---:|---:|---:|']
    for label,r in s['teacher_counts'].items():lines.append(f"|{label}|{r['total']}|{r['soil_obtained']}|{r['solphy_match']}|{r['any_mean']}|")
    lines += ['', '空間結合はwithin。同じコードの重複は1行にし、異なる分類への複数一致はUNKNOWNとする設計。境界上等で取得不能となる場合もあります。',
        f"複数一致の追加行数={s['overlap_extra_matches']}、異なる分類への曖昧一致試料数={s['ambiguous_points']}。", '',
        '|土壌分類|実測|疑似|','|---|---:|---:|']
    for r in counts.itertuples(index=False):lines.append(f'|{r.soil_category}|{r.measured}|{r.predicted}|')
    lines += ['', '## 空間分割の未知カテゴリ', '',
        '実測のみと実測＋疑似の学習集合を別々に集計。テストは同一の実測のみ。UNKNOWNも1カテゴリとして比較し、欠損件数を別列で保存しています。', '',
        '|seed|教師条件|テスト数|未知カテゴリ試料数|未知率|','|---|---|---:|---:|---:|']
    for r in cv.itertuples(index=False):lines.append(f'|{r.seed}|{r.condition}|{r.test_count}|{r.unseen_count}|{r.unseen_rate:.1%}|')
    lines += ['', '学習・テスト・テストのみのカテゴリの全一覧はspatial_cv_categories.csvに保存。', '',
        '## 利用可否の判断', '',
        '土壌図のカテゴリは座標から取得できるため、欠損・未知カテゴリを明示したカテゴリ説明変数として比較可能です。ただし未知カテゴリ率は分割ごとに異なり、取得不能地域の外挿には制約があります。',
        'SolphyJはSSGコードでの結合が可能な範囲に限り候補として利用できます。単一のSSGには土地利用5区分×層位6区分の行があり、SSGだけでは物性値は一意に決まりません。土地利用・層位を選ぶか、集計する規則を決めるまでモデルへ値を入力すべきではありません。Horizonの番号をボーリング深度へ自動対応させていません。',
        '土壌分類取得と物性値取得を区別し、SSG別の既存統計が深部試料の物性を代表するとは仮定していません。対応できないコードについて推測で補完していません。', '',
        '再実行：`python -m module.soil_inventory_audit` → `python -m module.soil_inventory_report`。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
