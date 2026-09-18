"""Refit the selected predicted-N condition for full-grid deployment."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from module.paths import TRAINING
from module.gamma_features import model_frame
from module.n_estimation import fit_predict, exclude_queries, spatial_keys, predict_ensemble
from module.model_reporting import save_model_reports
from module.xgb_common import build_preprocessor, build_regressor


def train_selected(config, run):
    options=config['n_comparison'];validation=config['validation']
    mode=validation['mode']
    frame=pd.read_csv(TRAINING/'gamma/measurements.csv')
    required={'n_measured','comparison_hole_id','surface_z','depth','gamma_wet','gamma_dry'}
    if required-set(frame):
        raise ValueError('Rerun gamma preparation before predicted_n deployment')
    frame=frame.loc[frame.n_measured.notna() & frame[['gamma_wet','gamma_dry']].notna().any(axis=1)].copy()
    if frame.empty: raise ValueError('No primary density samples for predicted_n deployment')
    frame['boring_id']=frame.comparison_hole_id
    frame['n_value_elevation']=frame.surface_z-frame.depth
    spt=pd.read_csv(TRAINING/'gamma/n_spt.csv')
    numeric=options['numeric_features'];categorical=options['categorical_features']
    groups=spatial_keys(frame,validation['block_size_m']) if mode=='spatial' else frame.boring_id
    folds=min(options['cross_fit_folds'],groups.nunique())
    if folds<2: raise ValueError('Too few primary groups for final gamma cross-fitting')
    predicted=[]
    for fold,(_,index) in enumerate(GroupKFold(n_splits=folds).split(frame,groups=groups)):
        query=frame.iloc[index].copy()
        source=exclude_queries(spt,query,validation,options['xy_tolerance_m'])
        predicted.append(fit_predict(source,query,validation,options,numeric,categorical,
                                    options['xgb_parameters'],run/f'n_models/refit_fold_{fold}'))
    data=pd.concat(predicted,ignore_index=True)
    data['n_input']=data.n_predicted
    data.to_csv(run/'final_training_n_features.csv',index=False)
    result={}
    features=[*config['numeric_features'],'n_input']
    for kind in ('wet','dry'):
        # Tree count comes from the primary-mode validation, never from test R².
        evaluation_path=run/f'n_comparison/{mode}/{kind}/predicted_n/metrics.json'
        evaluation=json.loads(evaluation_path.read_text())
        if evaluation['status']!='complete': raise ValueError(f'Incomplete predicted_n evaluation: {kind}')
        selected=data.loc[data['gamma_'+kind].notna()].copy()
        if selected.empty: raise ValueError(f'No {kind} primary samples')
        transform=build_preprocessor(features,config['categorical_features'])
        x=transform.fit_transform(model_frame(selected,features,config['categorical_features']))
        params=dict(config['xgb_parameters'],n_estimators=evaluation['best_n_estimators'])
        params.pop('early_stopping_rounds',None)
        model=build_regressor(params);model.fit(x,selected['gamma_'+kind],verbose=False)
        output=run/'models'/kind;output.mkdir(parents=True,exist_ok=True)
        model.save_model(output/'model.json');joblib.dump(transform,output/'preprocessor.joblib')
        save_model_reports(model,transform,features,config['categorical_features'],selected['gamma_'+kind],output,
                           'Unit weight gamma (kN/m³)',population='all primary final fitting records',density_gravity=config['gravity_m_s2'])
        metrics={'distribution_model':'predicted_n','numeric_features':features,
                 'categorical_features':config['categorical_features'],
                 'final_fit_rows':len(selected),'final_fit_holes':int(selected.boring_id.nunique()),
                 'training_population':'all primary samples only; no supplemental samples',
                 'n_training_method':'out-of-hole/area cross-fitted ensemble mean capped at 50',
                 'training_depth_range_m':[float(selected.depth.min()),float(selected.depth.max())],
                 'target_unit':'kN/m3','xgb_parameters':params,
                 'evaluation_reference':str(evaluation_path.relative_to(run)),
                 'evaluation':evaluation,'evaluation_is_for_holdout_model_not_final_refit':True}
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        pd.DataFrame({'feature':transform.get_feature_names_out(),'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(output/'feature_importance.csv',index=False)
        # Retain the held-out scatter and its R² beside the final artifacts, with provenance.
        import shutil
        for filename in ('observed_vs_predicted.png','test_predictions.csv'):
            shutil.copyfile(evaluation_path.parent/filename,output/filename)
        result[kind]=(model,transform,metrics)
    # Independent upstream fit for grid inference. Calibration rows stay excluded
    # from fitting so the stored empirical interval is consistent with this ensemble.
    upstream=run/'n_models/grid'
    fit_predict(spt,frame.head(1),validation,options,numeric,categorical,
                options['xgb_parameters'],upstream)
    from xgboost import XGBRegressor
    models=[]
    for seed in options['ensemble_seeds']:
        model=XGBRegressor(n_jobs=options['xgb_parameters'].get('n_jobs',4));model.load_model(upstream/f'model_seed_{seed}.json');models.append(model)
    artifact=(models,joblib.load(upstream/'preprocessor.joblib'),
              json.loads((upstream/'metrics.json').read_text()))
    return result,artifact


def add_predicted_n(frame,artifact):
    models,transform,metrics=artifact
    frame=frame.copy()
    frame['n_value_elevation']=frame.surface_z-frame.depth
    samples=predict_ensemble(models,transform,frame,metrics['numeric_features'],metrics['categorical_features'])
    if not np.isfinite(samples).all():
        raise ValueError('Non-finite N predictions; gamma inference stopped')
    frame['n_input']=samples.mean(axis=0)
    frame['n_ensemble_std']=samples.std(axis=0,ddof=1)
    frame['n_lower']=np.clip(frame.n_input-metrics['interval_radius'],0.,50.)
    frame['n_upper']=np.clip(frame.n_input+metrics['interval_radius'],0.,50.)
    return frame
