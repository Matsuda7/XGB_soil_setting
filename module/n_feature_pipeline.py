"""Shared leakage-safe N features for a downstream soil-property model."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold
from module.n_estimation import fit_predict, exclude_queries, spatial_keys


def crossfit_features(frame,spt,config,output,validation=None):
    options=config['n_comparison'];validation=validation or config['validation']
    if validation['mode']=='row': raise ValueError('Predicted N features require borehole/spatial splitting')
    groups=spatial_keys(frame,validation['block_size_m']) if validation['mode']=='spatial' else frame.boring_id
    folds=min(options['cross_fit_folds'],groups.nunique())
    if folds<2: raise ValueError('Too few groups for cross-fitted N features')
    result=[]
    for fold,(_,index) in enumerate(GroupKFold(n_splits=folds).split(frame,groups=groups)):
        query=frame.iloc[index].copy()
        source=exclude_queries(spt,query,validation,options['xy_tolerance_m'])
        values=fit_predict(source,query,validation,options,options['numeric_features'],options['categorical_features'],
                           options['xgb_parameters'],output/f'fold_{fold}')
        values['n_input']=values.n_predicted;result.append(values)
    return pd.concat(result,ignore_index=True)


def evaluation_features(parts,spt,config,output,validation):
    train,val,test=parts;options=config['n_comparison']
    heldout=pd.concat([val,test],ignore_index=True)
    source=exclude_queries(spt,heldout,validation,options['xy_tolerance_m'])
    predicted=fit_predict(source,heldout,validation,options,options['numeric_features'],options['categorical_features'],
                          options['xgb_parameters'],output/'heldout')
    predicted['n_input']=predicted.n_predicted
    val=predicted.loc[predicted.record_id.isin(val.record_id)].copy()
    test=predicted.loc[predicted.record_id.isin(test.record_id)].copy()
    train=crossfit_features(train,source,config,output/'training',validation)
    for name,part in zip(('train','validation','test'),(train,val,test)):
        part.to_csv(output/f'{name}_n_features.csv',index=False)
    return train,val,test


def fit_grid_artifact(spt,query,config,output):
    options=config['n_comparison']
    fit_predict(spt,query.head(1),config['validation'],options,options['numeric_features'],options['categorical_features'],
                options['xgb_parameters'],output)
    from xgboost import XGBRegressor
    models=[]
    for seed in options['ensemble_seeds']:
        model=XGBRegressor(n_jobs=options['xgb_parameters'].get('n_jobs',4))
        model.load_model(output/f'model_seed_{seed}.json');models.append(model)
    return models,joblib.load(output/'preprocessor.joblib'),json.loads((output/'metrics.json').read_text())
