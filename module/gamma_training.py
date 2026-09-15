"""Separate wet/dry regressors using phi's shared training utilities."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module.paths import TRAINING
from module.validation import split_data, save_split
from module.gamma_config import KINDS, data_signature
from module.gamma_features import model_frame
from module.xgb_common import borehole_train_validation_test_split, build_preprocessor, build_regressor, regression_metrics


def fit_evaluation(parts, numeric, categorical, parameters):
    train,validation,test = parts
    preprocessor=build_preprocessor(numeric,categorical)
    xtrain=preprocessor.fit_transform(model_frame(train,numeric,categorical))
    xvalidation=preprocessor.transform(model_frame(validation,numeric,categorical))
    model=build_regressor(parameters)
    model.fit(xtrain,train.target,eval_set=[(xvalidation,validation.target)],verbose=False)
    predicted=model.predict(preprocessor.transform(model_frame(test,numeric,categorical)))
    return model,preprocessor,predicted


def train_variant(data,config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    required={'boring_id','target','x','y','depth',*config['numeric_features'],*config['categorical_features']}
    if required-set(data):
        raise ValueError(f'Missing gamma columns: {sorted(required-set(data))}')
    if data[['boring_id','target','x','y','depth']].isna().any(axis=None) or not np.isfinite(data.target).all() or (data.target<=0).any():
        raise ValueError('Gamma training requires IDs, coordinates, depth and positive finite targets')
    parts=split_data(data, config['validation'], target_column='target')
    save_split(parts, output/'validation_split.csv')
    train,validation,test=parts
    numeric,categorical=config['numeric_features'],config['categorical_features']
    parameters=dict(config['xgb_parameters'])
    model,preprocessor,prediction=fit_evaluation(parts,numeric,categorical,parameters)
    best=int(model.best_iteration)+1
    spatial_metrics = None
    validation_config = config['validation']
    if validation_config['additional_spatial'] and validation_config['mode'] != 'spatial':
        spatial_parts = split_data(data, dict(validation_config, mode='spatial'), target_column='target')
        spatial_model, spatial_preprocessor, spatial_prediction = fit_evaluation(spatial_parts, numeric, categorical, parameters)
        spatial_metrics = regression_metrics(spatial_parts[2].target, spatial_prediction)
        save_split(spatial_parts, output/'spatial_split.csv')
        spatial_test = spatial_parts[2][['boring_id','x','y','depth','target']].copy()
        spatial_test['predicted_gamma_kn_m3'] = spatial_prediction
        spatial_test.to_csv(output/'spatial_test_predictions.csv', index=False)
        (output/'spatial_metrics.json').write_text(json.dumps({'test':spatial_metrics, 'validation':dict(validation_config, mode='spatial')}, indent=2)+'\n')
        del spatial_model, spatial_preprocessor
    comparison=[{'features':'median_of_training_targets',**regression_metrics(test.target,np.full(len(test),train.target.median()))}]
    baseline,baseline_preprocessor,baseline_prediction=fit_evaluation(parts,['x','y','depth'],[],parameters)
    comparison.append({'features':'position_and_depth',**regression_metrics(test.target,baseline_prediction)})
    comparison.append({'features':'configured_geographic_features',**regression_metrics(test.target,prediction)})
    pd.DataFrame(comparison).to_csv(output/'feature_comparison.csv',index=False)
    predictions=test[['boring_id','x','y','depth','target']].copy()
    predictions['predicted_gamma_kn_m3']=prediction
    predictions['residual']=predictions.target-prediction
    predictions.to_csv(output/'test_predictions.csv',index=False)
    split=pd.concat([part[['boring_id']].drop_duplicates().assign(split=name) for name,part in zip(('train','validation','test'),parts)])
    split.to_csv(output/'borehole_split.csv',index=False)
    # Only validation chooses the tree count. The configured feature set is fixed in advance.
    final_preprocessor=build_preprocessor(numeric,categorical)
    xall=final_preprocessor.fit_transform(model_frame(data,numeric,categorical))
    final_parameters=dict(parameters);final_parameters.pop('early_stopping_rounds',None);final_parameters['n_estimators']=best
    final_model=build_regressor(final_parameters);final_model.fit(xall,data.target,verbose=False)
    final_model.save_model(output/'model.json');joblib.dump(final_preprocessor,output/'preprocessor.joblib')
    importance=pd.DataFrame({'feature':final_preprocessor.get_feature_names_out(),'importance':final_model.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(output/'feature_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(test.target,prediction,s=24,alpha=.75)
    low=min(test.target.min(),prediction.min());high=max(test.target.max(),prediction.max())
    ax.plot([low,high],[low,high],'k--',lw=1)
    ax.set(xlabel='Observed gamma [kN/m3]',ylabel='Predicted gamma [kN/m3]',title=f"Held-out test predictions ({config['validation']['mode']})")
    test_metrics = regression_metrics(test.target, prediction)
    ax.text(0.04, 0.96, f"R² = {test_metrics['r2']:.4f}", transform=ax.transAxes,
            ha='left', va='top', bbox={'facecolor':'white', 'alpha':0.8, 'edgecolor':'none'})
    fig.tight_layout();fig.savefig(output/'observed_vs_predicted.png',dpi=160);plt.close(fig)
    metrics={'test':regression_metrics(test.target,prediction),'best_n_estimators':best,
        'numeric_features':numeric,'categorical_features':categorical,'xgb_parameters':parameters,
        'rows':{name:len(part) for name,part in zip(('train','validation','test'),parts)},
        'boreholes':{name:int(part.boring_id.nunique()) for name,part in zip(('train','validation','test'),parts)},
        'final_fit_rows':len(data),'target_unit':'kN/m3','target_range':[float(data.target.min()),float(data.target.max())],
        'training_depth_range_m':[float(data.depth.min()),float(data.depth.max())],
        'feature_selection':'configured in advance; comparison does not automatically choose features',
        'split':config['validation'],'additional_spatial':spatial_metrics,'preparation_signature':data_signature(config)}
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(f"gamma model {output.name}: rows={len(data)}, trees={best}, test={metrics['test']}",flush=True)
    return final_model,final_preprocessor,metrics


def train_both(config,output):
    summary=json.loads((TRAINING/'gamma/preparation_summary.json').read_text())
    if summary['signature']!=data_signature(config):
        raise ValueError('Gamma geographic preparation settings changed; rerun the prepare stage')
    return {kind:train_variant(pd.read_csv(TRAINING/f'gamma/{kind}.csv'),config,Path(output)/kind) for kind in KINDS}
