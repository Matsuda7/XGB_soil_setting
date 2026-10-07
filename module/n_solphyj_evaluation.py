"""Paired N-value evaluation with SSG representative texture features."""
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from module.paths import project_path
from module.phi_training import load_dataset, XGB_PARAMETERS
from module.xgb_common import load_model_config, build_preprocessor, build_regressor, prepare_feature_frame, regression_metrics
from module.validation import split_data

PROPERTIES = ['representative_total_sand', 'representative_silt', 'representative_clay']


def main():
    out = project_path('n_test/solphyj_evaluation_borehole')
    out.mkdir(parents=True, exist_ok=False)
    numeric, categorical = load_model_config(project_path('config/phi/model.jsonc'))
    numeric = [c for c in numeric if c not in PROPERTIES]
    frame = load_dataset(project_path('data/training/phi/model_dataset.csv'), numeric, categorical)
    frame['row_id'] = np.arange(len(frame))
    crs = json.loads(project_path('config/spatial.json').read_text())['grid_crs']
    soil = gpd.read_file('zip://' + str(project_path('data/raw/soil_inventory/石川_20.zip')), encoding='cp932').to_crs(crs)
    points = gpd.GeoDataFrame(frame[['row_id', 'x', 'y']], geometry=gpd.points_from_xy(frame.x, frame.y), crs=crs)
    joined = gpd.sjoin(points, soil[['SSerGrCD', 'geometry']], how='left', predicate='within')
    ambiguous = joined.groupby('row_id').SSerGrCD.nunique().gt(1)
    joined = joined.sort_values(['row_id', 'index_right']).drop_duplicates('row_id')
    joined.loc[joined.row_id.isin(ambiguous[ambiguous].index), 'SSerGrCD'] = pd.NA
    frame = frame.merge(joined[['row_id', 'SSerGrCD']], on='row_id', validate='one_to_one', sort=False)
    props = pd.read_csv(project_path('gamma_test/solphyj_representative_soil/ssg_representative_properties.csv'))
    frame = frame.merge(props[['SSG code', *PROPERTIES]], left_on='SSerGrCD', right_on='SSG code', how='left', validate='many_to_one', sort=False)
    frame['solphyj_available'] = frame[PROPERTIES].notna().all(axis=1)
    frame[['row_id', 'boring_id', 'measurement_no', 'SSerGrCD', *PROPERTIES]].to_csv(out/'soil_features.csv', index=False)
    settings = dict(numeric=numeric, categorical=categorical, added=PROPERTIES, parameters=XGB_PARAMETERS,
                    input='data/training/phi/model_dataset.csv', total=len(frame), soil_available=int(frame.solphyj_available.sum()),
                    ambiguous_rows=int(ambiguous.sum()), missing_policy='Existing numeric median imputation retained; added SolphyJ numeric NaN passed directly to XGBoost',
                    split_seeds=[42,43,44,45,46])
    (out/'settings.json').write_text(json.dumps(settings, indent=2))
    rows=[]
    # Repeated borehole holdout assessment.
    for mode, seed in [('borehole', s) for s in settings['split_seeds']]:
        split_config=dict(mode=mode, test_size=.2, validation_size=.2, random_state=seed)
        parts=split_data(frame, split_config)
        train,val,test=parts
        assert not (set(train._validation_group) & set(val._validation_group) or set(train._validation_group) & set(test._validation_group) or set(val._validation_group) & set(test._validation_group))
        pd.concat([p[['row_id','boring_id','_validation_group']].assign(split=s) for p,s in zip(parts,['train','validation','test'])]).to_csv(out/f'{mode}_{seed}_split.csv',index=False)
        for name, extra in [('baseline',[]),('with_solphyj',PROPERTIES)]:
            pre=build_preprocessor(numeric,categorical)
            if extra:
                pre.transformers.append(('solphyj','passthrough',extra))
            transformed=[prepare_feature_frame(p,numeric+extra,categorical) for p in parts]
            xt=pre.fit_transform(transformed[0]); xv=pre.transform(transformed[1]); xe=pre.transform(transformed[2])
            model=build_regressor(XGB_PARAMETERS)
            model.fit(xt,train.n_value,eval_set=[(xv,val.n_value)],verbose=False)
            pred=np.clip(model.predict(xe),0,50)
            rows.append(dict(mode=mode,seed=seed,model=name,train_count=len(train),validation_count=len(val),test_count=len(test),
                test_solphyj_available=int(test.solphyj_available.sum()),best_iteration=int(model.best_iteration),**regression_metrics(test.n_value,pred)))
            test[['row_id','boring_id','n_value']].assign(prediction=pred).to_csv(out/f'{mode}_{seed}_{name}_predictions.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(mode,seed,name,rows[-1]['r2'],flush=True)
    scores=pd.DataFrame(rows)
    summary=scores.groupby(['mode','model'])[['r2','rmse','mae']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    (out/'README.md').write_text('# N値へのSolphyJ追加評価\n\n'
        '既存N値モデルとSSG代表Sand/Silt/Clay追加モデルを同一データ・分割で比較。N値は既存の上限50、予測も0～50。'
        '主評価は10 km空間分割seed 42～46。参考として既存設定のボーリング孔分割seed 42も評価。'
        '既存特徴量の中央値補完・カテゴリエンコードは学習データのみで適合。追加SolphyJは欠損をNaNのまま入力し行を除外しない。'
        '土壌コード完全一致で結合し、重複ポリゴンでコードが相違した場合は欠損とする。SolphyJは表層代表物性であり深度の実粒度ではない。'
        '最終モデル保存・全域予測は行っていない。\n\n'
        f'教師データ{len(frame)}件、SolphyJ取得{int(frame.solphyj_available.sum())}件。\n\n```text\n'+summary.to_string()+'\n```\n'
        '\n再実行: python -m module.n_solphyj_evaluation（既存出力がある場合は停止）\n')
    print(summary.to_string())


if __name__=='__main__':
    main()
