"""Geology audit and paired evaluation using the Step 2 evaluation engine."""
import argparse
import json
import geopandas as gpd
import pandas as pd
from module.paths import project_path
from module.spatial_features import GEOLOGY_DIR
from module.gamma_stage2_terrain import run


def attach_geology(frame, out):
    paths = sorted(GEOLOGY_DIR.rglob('*_poly.shp'))
    parts = [gpd.read_file(path) for path in paths]
    if not parts or any(p.crs != parts[0].crs for p in parts):
        raise ValueError('Missing geology or inconsistent CRS')
    geology = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    pd.DataFrame([dict(attribute=c, dtype=str(geology[c].dtype), unique_count=geology[c].nunique())
                  for c in geology.columns if c!='geometry']).to_csv(out/'attributes.csv', index=False)
    points = gpd.GeoDataFrame(frame[['record_id','label_source','x','y']].copy(),
        geometry=gpd.points_from_xy(frame.x, frame.y), crs='EPSG:6675').to_crs(geology.crs)
    joined = gpd.sjoin(points, geology[['symbol','ser','geometry']], how='left', predicate='within')
    overlaps = len(joined)-len(frame)
    joined = joined.sort_values(['record_id','index_right'], kind='stable').drop_duplicates('record_id')
    joined['geology'] = joined.symbol.astype('string').fillna('UNKNOWN')
    attributes = joined[['record_id','symbol','ser','geology']]
    attributes.to_csv(out/'geology_features.csv', index=False)
    counts = pd.crosstab(joined.geology, joined.label_source).reindex(columns=['measured','predicted'], fill_value=0)
    counts = counts.reindex(sorted(set(geology.symbol.dropna()) | {'UNKNOWN'}), fill_value=0)
    counts['total'] = counts.sum(axis=1)
    counts.to_csv(out/'category_counts.csv')
    metadata = dict(files=[str(p) for p in paths], crs=str(geology.crs), selected_attribute='symbol',
        map_categories=int(geology.symbol.nunique()), data_categories=int(joined.symbol.nunique()),
        overlap_extra_matches=overlaps, spatial_join='within; lowest polygon index wins overlaps; unmatched -> UNKNOWN',
        encoding='OneHotEncoder(handle_unknown=ignore), fitted on each second-stage training set only',
        unseen_category='all zeros for geology one-hot columns; no category ordering',
        missing=counts.loc['UNKNOWN'].to_dict(),
        no_measured=counts.index[(counts.measured==0)&(counts.index!='UNKNOWN')].tolist(),
        pseudo_only=counts.index[(counts.measured==0)&(counts.predicted>0)].tolist())
    (out/'geology_audit.json').write_text(json.dumps(metadata, indent=2)+'\n')
    result = frame.drop(columns=['geology'], errors='ignore').merge(attributes[['record_id','geology']], on='record_id', how='left', validate='one_to_one', sort=False)
    if result.geology.isna().any():
        raise ValueError('Lost record during geology join')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/gamma/stage2_geology.json')
    args = parser.parse_args()
    settings = json.loads(project_path(args.config).read_text())
    out = project_path(settings['output']); out.mkdir(exist_ok=True, parents=True)
    scores = run(settings, lambda frame: attach_geology(frame, out))
    deltas = pd.read_csv(out/'deltas.csv')
    stability = deltas.groupby(['condition','model']).agg(
        r2_improved=('delta_r2', lambda s:int((s>0).sum())),
        rmse_improved=('delta_rmse', lambda s:int((s<0).sum())),
        mae_improved=('delta_mae', lambda s:int((s<0).sum())))
    stability.to_csv(out/'stability.csv')
    # Isolate added geology relative to matching terrain models as well as G0.
    step2 = pd.read_csv(project_path('gamma_test/stage2_terrain/metrics.csv'))
    matched = scores.loc[scores.model.isin(['G2','G3'])].copy()
    matched['terrain_model'] = matched.model.map({'G2':'M2','G3':'M3'})
    matched = matched.merge(step2, left_on=['seed','condition','terrain_model'],
        right_on=['seed','condition','model'], suffixes=('', '_terrain'), validate='one_to_one')
    for metric in ['r2','rmse','mae']:
        matched['delta_'+metric] = matched[metric]-matched[metric+'_terrain']
    matched.to_csv(out/'matched_terrain_deltas.csv', index=False)
    summary = scores.groupby(['condition','model'])[['r2','rmse','mae']].agg(['mean','std'])
    ds = deltas.groupby(['condition','model'])[['delta_r2','delta_rmse','delta_mae']].agg(['mean','std'])
    lines = ['# Step 3：第二段階の地質情報追加', '',
        'G0=x,y,elevation。G1=G0+地質。G2=G0+curvature+地質。G3=G0+slope+curvature+地質。',
        '地質属性はsymbol、学習部分だけでOne-hotエンコード。serは連続値としてもカテゴリとしても併用しません。未知カテゴリは全ゼロ、欠損はUNKNOWN。カテゴリ統合・整数順序付けは行っていません。',
        'Step 1と同じ孔単位分割seed 42〜46をID・順序まで再利用。第一段階は各seedで再学習し、検証・テスト実測を使用しません。Step 2の監査とG0のStep 1再現チェックが全seedで通過。重み・XGBoost設定は変更していません。', '',
        '|条件|モデル|R² 平均±SD|RMSE 平均±SD|MAE 平均±SD|','|---|---|---:|---:|---:|']
    for (condition, model), row in summary.iterrows():
        lines.append('|'+ '|'.join([condition,model,*[f"{row[(m,'mean')]:.4f} ± {row[(m,'std')]:.4f}" for m in ['r2','rmse','mae']]])+'|')
    lines += ['', '## G0との差（変更後−G0）', '', '|条件|モデル|ΔR² 平均±SD|ΔRMSE 平均±SD|ΔMAE 平均±SD|', '|---|---|---:|---:|---:|']
    for (condition, model), row in ds.iterrows():
        lines.append('|'+ '|'.join([condition,model,*[f"{row[(m,'mean')]:+.4f} ± {row[(m,'std')]:.4f}" for m in ['delta_r2','delta_rmse','delta_mae']]])+'|')
    lines += ['', '## 改善した分割数（全5回）', '', '|条件|モデル|R²|RMSE|MAE|', '|---|---|---:|---:|---:|']
    for (condition, model), r in stability.iterrows():
        lines.append(f'|{condition}|{model}|{r.r2_improved}|{r.rmse_improved}|{r.mae_improved}|')
    lines += ['', '## 地質データ監査', '',
        '属性はsymbolとser。地図全体101区分、教師データ内16区分＋UNKNOWN。実測がない地図カテゴリは87種類（85種類は教師データ自体がない）。疑似のみはN1_sbs（4件）とN3_som（3件）。取得不能は実測10件・疑似52件。',
        '実測1〜4件のカテゴリがあり、疎なカテゴリの汎化には注意が必要です。カテゴリ別全件数はcategory_counts.csv、属性一覧はattributes.csv、地図全体で実測のないカテゴリ一覧はgeology_audit.jsonに保存。',
        '地図との結合は既存処理と同じwithin。境界上等はUNKNOWNになり得ます。今回複数ポリゴン一致は0件。',
        '地形計算はStep 2と同一（10 m DEM、傾斜は度、曲率はラプラシアンm⁻¹）。設定と欠損処理はsettings.jsonに記録。', '',
        '## 保存ファイル', '',
        'metrics.csv：各seed。summary.csv：平均・標本標準偏差。deltas.csv / delta_summary.csv：G0との差。stability.csv：改善回数。matched_terrain_deltas.csv：G2対M2、G3対M3の比較。unseen_categories.csv：学習で未知の検証・テストカテゴリ。comparison.png：比較図。',
        '各borehole_*に分割ID・第一段階監査・再生成疑似値・予測値を保存。最終モデル保存・全域予測はしていません。',
        '5回は反復ホールドアウトで、SDはddof=1。過去の説明変数選定に使用した実測を再利用しており、独立外部検証ではありません。', '',
        '再実行：`python -m module.gamma_stage2_geology`']
    lines += ['', '## 採用根拠の判断', '',
        '地質を比較候補として残す限定的な根拠は得られました。G1は両教師条件でG0よりR²・RMSE・MAEが5回中4回改善しました。ただし平均R²は依然負です。',
        '実測のみのG2は平均R²=0.0078（SD=0.0549）で最良ですが、Step 2のM2に対しては平均R²差+0.0227、改善は5回中2回のみです。G0との差には曲率の効果も含まれるため、G2の改善全体を地質だけの効果とは解釈できません。',
        '疑似追加のG1は平均R²=-0.3888でG0より改善しますが、実測のみより低精度のままです。地質追加が疑似γ追加の問題を解消したとは言えません。',
        'seed 42では58テスト試料中51件が学習にないカテゴリで、One-hotの地質部分は全ゼロになります。未知地域の地質が学習地域にない場合の一般化は、今回のエンコードでは限定されます。',
        '現時点では地質を正式採用して安定した高精度が得られるという強い根拠は不十分です。平均の改善と分割依存性の両方を記録し、独立検証を残します。']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(stability.to_string())


if __name__ == '__main__':
    main()
