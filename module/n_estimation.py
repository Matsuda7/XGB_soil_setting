"""Nested held-out SPT estimation for gamma comparisons, with uncertainty diagnostics."""
import json
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from module.validation import split_data, save_split
from module.gamma_features import model_frame
from module.model_reporting import save_model_reports
from module.xgb_common import build_preprocessor, build_regressor, regression_metrics

RELIABILITY_COLUMNS = ['n_ensemble_std','n_interval_width','n_nearest_hole_distance_m',
                       'n_nearby_hole_count','n_nearest_hole_depth_gap_m','n_depth_extrapolation']


def spatial_keys(frame, size):
    xy = frame.groupby('boring_id')[['x','y']].transform('median')
    cells = np.floor(xy / size).astype('int64')
    return cells.x.astype(str)+':'+cells.y.astype(str)


def exclude_queries(spt, heldout, validation, tolerance):
    if heldout.empty:
        return spt.copy()
    holes = spt[['boring_id','x','y']].drop_duplicates('boring_id')
    distances, _ = cKDTree(heldout[['x','y']].to_numpy()).query(holes[['x','y']].to_numpy())
    excluded = set(holes.loc[distances <= tolerance, 'boring_id'])
    # Also use recorded correspondence in case source coordinates were rounded.
    if 'spt_boring_id' in heldout:
        excluded.update(heldout.spt_boring_id.dropna().astype(str))
    allowed = ~spt.boring_id.isin(excluded)
    if validation['mode'] == 'spatial':
        forbidden = set(spatial_keys(heldout, validation['block_size_m']))
        allowed &= ~spatial_keys(spt, validation['block_size_m']).isin(forbidden)
    return spt.loc[allowed].copy()


def predict_ensemble(models, preprocessor, frame, numeric, categorical):
    values = preprocessor.transform(model_frame(frame, numeric, categorical))
    return np.stack([np.clip(model.predict(values), 0., 50.) for model in models])


def fit_predict(spt, query, validation, options, numeric, categorical, parameters, output):
    output.mkdir(parents=True, exist_ok=True)
    from module.n_matching import location_groups
    spt = spt.copy()
    spt['source_boring_id'] = spt.boring_id
    spt['boring_id'] = location_groups(spt, options['xy_tolerance_m'])
    internal = dict(validation, additional_spatial=False)
    train, tuning, calibration = split_data(spt, internal)
    save_split((train,tuning,calibration), output/'split.csv')
    spt[['source_boring_id','boring_id','x','y']].drop_duplicates().to_csv(output/'source_hole_groups.csv', index=False)
    transform = build_preprocessor(numeric,categorical)
    xt = transform.fit_transform(model_frame(train,numeric,categorical))
    xv = transform.transform(model_frame(tuning,numeric,categorical))
    tuning_model = build_regressor(parameters)
    tuning_model.fit(xt,train.n_value,eval_set=[(xv,tuning.n_value)],verbose=False)
    save_model_reports(tuning_model,transform,numeric,categorical,train.n_value,output/'tuning','SPT N-value')
    trees = int(tuning_model.best_iteration)+1
    source = pd.concat([train,tuning],ignore_index=True)
    transform = build_preprocessor(numeric,categorical)
    xs = transform.fit_transform(model_frame(source,numeric,categorical))
    models=[]
    for seed in options['ensemble_seeds']:
        params=dict(parameters, random_state=seed, n_estimators=trees)
        params.pop('early_stopping_rounds',None)
        model=build_regressor(params);model.fit(xs,source.n_value,verbose=False)
        model.save_model(output/f'model_seed_{seed}.json');models.append(model)
        save_model_reports(model,transform,numeric,categorical,source.n_value,output/f'reports/seed_{seed}',
                           'SPT N-value',population='N-model train plus tuning; calibration excluded')
    joblib.dump(transform, output/'preprocessor.joblib')
    cal_samples=predict_ensemble(models,transform,calibration,numeric,categorical)
    cal_prediction=cal_samples.mean(axis=0)
    errors=np.abs(calibration.n_value.to_numpy()-cal_prediction)
    # Empirical residual interval, not a coverage guarantee for correlated spatial data.
    radius=float(np.quantile(errors,options['interval_coverage'],method='higher'))
    cal_table=calibration[['boring_id','x','y','depth','n_value']].copy()
    cal_table['n_predicted']=cal_prediction
    cal_table.to_csv(output/'calibration_predictions.csv',index=False)
    samples=predict_ensemble(models,transform,query,numeric,categorical)
    result=query.copy()
    result['n_predicted']=samples.mean(axis=0)
    result['n_ensemble_std']=samples.std(axis=0,ddof=1)
    result['n_lower']=np.clip(result.n_predicted-radius,0.,50.)
    result['n_upper']=np.clip(result.n_predicted+radius,0.,50.)
    result['n_interval_width']=result.n_upper-result.n_lower
    holes=source[['boring_id','x','y']].drop_duplicates('boring_id').reset_index(drop=True)
    tree=cKDTree(holes[['x','y']].to_numpy())
    distance, index=tree.query(query[['x','y']].to_numpy())
    result['n_nearest_hole_distance_m']=distance
    result['n_nearby_hole_count']=tree.query_ball_point(query[['x','y']].to_numpy(),options['nearby_radius_m'],return_length=True)
    depths={k:g.depth.to_numpy() for k,g in source.groupby('boring_id')}
    result['n_nearest_hole_depth_gap_m']=[float(np.min(np.abs(depths[holes.iloc[i].boring_id]-d)))
                                        for i,d in zip(index,query.depth)]
    result['n_depth_extrapolation']=(~query.depth.between(source.depth.min(),source.depth.max())).astype(int)
    result['n_estimator_id']=output.name
    report={'trees':trees,'numeric_features':numeric,'categorical_features':categorical,
            'parameters':parameters,'ensemble_seeds':options['ensemble_seeds'],
            'source_rows':len(source),'source_holes':int(source.boring_id.nunique()),
            'calibration_rows':len(calibration),'calibration_metrics':regression_metrics(calibration.n_value,cal_prediction),
            'interval_radius':radius,'nominal_coverage':options['interval_coverage'],
            'interval_method':'absolute calibration residual quantile; no spatial coverage guarantee',
            'reliability_reference':'N-model final fitting sites (train+tuning); calibration and outer held-out sites excluded',
            'validation':internal}
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return result
