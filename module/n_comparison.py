"""Controlled gamma comparisons with common evaluation samples and nested SPT fits."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from module.paths import TRAINING
from module.validation import split_data, save_split
from module.gamma_features import model_frame
from module.xgb_common import build_preprocessor, build_regressor, regression_metrics
from module.n_estimation import exclude_queries, fit_predict, spatial_keys, RELIABILITY_COLUMNS


class InsufficientComparisonData(ValueError):
    pass


def fit_gamma(train, validation, numeric, categorical, parameters, weights=None):
    if len(train)<2 or len(validation)<2 or train.boring_id.nunique()<2:
        raise InsufficientComparisonData('At least two training holes and two validation rows are required')
    transform=build_preprocessor(numeric,categorical)
    xt=transform.fit_transform(model_frame(train,numeric,categorical))
    xv=transform.transform(model_frame(validation,numeric,categorical))
    model=build_regressor(parameters)
    model.fit(xt,train.target,eval_set=[(xv,validation.target)],sample_weight=weights,verbose=False)
    return model,transform


def save_evaluation(model,transform,test,numeric,categorical,output,gravity):
    output.mkdir(parents=True,exist_ok=True)
    predicted=model.predict(transform.transform(model_frame(test,numeric,categorical)))
    table=test.copy()
    table['predicted_gamma_kn_m3']=predicted
    table['observed_density_g_cm3']=test.target/gravity
    table['predicted_density_g_cm3']=predicted/gravity
    table.to_csv(output/'test_predictions.csv',index=False)
    metrics=regression_metrics(test.target,predicted)
    density_metrics=regression_metrics(test.target/gravity,predicted/gravity)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(table.observed_density_g_cm3,table.predicted_density_g_cm3,s=20,alpha=.7)
    lo=min(table.observed_density_g_cm3.min(),table.predicted_density_g_cm3.min())
    hi=max(table.observed_density_g_cm3.max(),table.predicted_density_g_cm3.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=1)
    ax.set(xlabel='Observed density (g/cm³)',ylabel='Predicted density (g/cm³)',title=output.name)
    ax.text(.04,.96,f"R² = {metrics['r2']:.4f}",transform=ax.transAxes,ha='left',va='top',
            bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    fig.tight_layout();fig.savefig(output/'observed_vs_predicted.png',dpi=160);plt.close(fig)
    return {'test_rows':len(test),'test_holes':int(test.boring_id.nunique()),
            'gamma_mae':metrics['mae'],'gamma_rmse':metrics['rmse'],'r2':metrics['r2'],
            'density_mae':density_metrics['mae'],'density_rmse':density_metrics['rmse'],
            'best_n_estimators':int(model.best_iteration)+1}


def comparison_for_kind(parts,kind,config,output):
    options=config['n_comparison'];numeric=config['numeric_features'];categorical=config['categorical_features']
    converted=[]
    for part in parts:
        frame=part.loc[part['gamma_'+kind].notna()].copy()
        frame['target']=frame['gamma_'+kind]
        converted.append(frame)
    train,validation,test=converted
    primary=[p.loc[p.n_measured.notna()].copy() for p in converted]
    train_p,val_p,test_p=primary
    if len(train_p)<2 or train_p.boring_id.nunique()<2 or len(val_p)<2 or len(test_p)<2:
        raise InsufficientComparisonData(f'{kind}: primary train/validation/test rows={list(map(len,primary))}')
    output.mkdir(parents=True,exist_ok=True)
    test_p[['record_id','source_boring_id','boring_id','x','y','depth','target']].to_csv(output/'common_test_samples.csv',index=False)
    rows=[]
    model_cache={}
    for scenario in options['scenarios']:
        supplemental=False
        if scenario=='no_n':
            fitting=train_p.copy();tuning=val_p.copy();evaluation=test_p.copy();features=numeric
        elif scenario in ('measured_n','measured_to_predicted_n'):
            fitting=train_p.assign(n_input=train_p.n_measured)
            tuning=val_p.assign(n_input=val_p.n_measured)
            evaluation=test_p.assign(n_input=test_p.n_measured if scenario=='measured_n' else test_p.n_predicted)
            features=[*numeric,'n_input']
        elif scenario in ('predicted_n','predicted_n_reliability'):
            fitting=train_p.assign(n_input=train_p.n_predicted)
            tuning=val_p.assign(n_input=val_p.n_predicted)
            evaluation=test_p.assign(n_input=test_p.n_predicted)
            features=[*numeric,'n_input']
            if scenario=='predicted_n_reliability': features+=RELIABILITY_COLUMNS
        elif scenario=='measured_plus_supplemental':
            fitting=train.assign(n_input=train.n_measured.fillna(train.n_predicted))
            tuning=val_p.assign(n_input=val_p.n_predicted)
            evaluation=test_p.assign(n_input=test_p.n_predicted)
            features=[*numeric,'n_input'];supplemental=True
        elif scenario=='no_n_with_supplemental':
            fitting=train.copy();tuning=val_p.copy();evaluation=test_p.copy();features=numeric;supplemental=True
        else:
            raise ValueError(f'Unknown N comparison scenario: {scenario}')
        weights=np.where(fitting.n_measured.notna(),1.,options['supplemental_weight']) if supplemental else None
        # Same fitted measured-N model for the two input substitutions.
        cache_key='measured_n' if scenario in ('measured_n','measured_to_predicted_n') else scenario
        if cache_key not in model_cache:
            model_cache[cache_key]=fit_gamma(fitting,tuning,features,categorical,config['xgb_parameters'],weights)
        model,transform=model_cache[cache_key]
        destination=output/scenario
        summary=save_evaluation(model,transform,evaluation,features,categorical,destination,config['gravity_m_s2'])
        fitting.assign(n_data_role=np.where(fitting.n_measured.notna(),'primary','supplemental')).to_csv(destination/'training_samples.csv',index=False)
        tuning.to_csv(destination/'validation_samples.csv',index=False)
        import joblib
        artifact = output/'models'/cache_key
        artifact.mkdir(parents=True, exist_ok=True)
        if not (artifact/'model.json').exists():
            model.save_model(artifact/'model.json');joblib.dump(transform,artifact/'preprocessor.joblib')
        row={'scenario':scenario,'model_directory':str(artifact.relative_to(output)), 'status':'complete','training_rows':len(fitting),
             'primary_training_rows':int(fitting.n_measured.notna().sum()),
             'supplemental_training_rows':int(fitting.n_measured.isna().sum()),
             'validation_rows':len(tuning),'numeric_features':features,**summary}
        (destination/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
    del model_cache
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    predicted=test_p.n_predicted.to_numpy();measured=test_p.n_measured.to_numpy()
    reliability={'n_test':regression_metrics(measured,predicted),
        'interval_coverage_on_common_test':float(np.mean((measured>=test_p.n_lower)&(measured<=test_p.n_upper))),
        'mean_interval_width':float(test_p.n_interval_width.mean()),
        'note':'Reference measured N may be an interval mean; diagnostic coverage, not a guarantee.'}
    (output/'n_reliability_test.json').write_text(json.dumps(reliability,indent=2)+'\n')
    return rows


def run_comparison(config,output):
    options=config['n_comparison'];output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    import shutil
    for name in ('n_matching.csv','n_matching_summary.json'):
        shutil.copyfile(TRAINING/'gamma'/name, output/name)
    frame=pd.read_csv(TRAINING/'gamma/measurements.csv')
    spt=pd.read_csv(TRAINING/'gamma/n_spt.csv')
    frame=frame.loc[frame[['gamma_wet','gamma_dry']].notna().any(axis=1)].copy()
    required={'n_measured','comparison_hole_id','source_boring_id','record_id'}
    if required-set(frame): raise ValueError('Rerun gamma preparation to add N correspondence columns')
    frame['boring_id']=frame.comparison_hole_id
    frame['n_value_elevation']=frame.surface_z-frame.depth
    numeric=options['numeric_features'];categorical=options['categorical_features']
    for name,table in [('SPT',spt),('density',frame)]:
        if set(numeric+categorical)-set(table): raise ValueError(f'{name}: unavailable N model features')
    modes=[config['validation']['mode']]
    if config['validation']['additional_spatial'] and 'spatial' not in modes: modes.append('spatial')
    all_rows=[];reports=[]
    for mode in modes:
        if mode=='row': raise ValueError('N comparison requires borehole or spatial validation; disable comparison for row validation')
        directory=output/mode;directory.mkdir(exist_ok=True)
        validation=dict(config['validation'],mode=mode)
        if frame.boring_id.nunique()<5:
            reports.append({'mode':mode,'status':'insufficient_data','reason':'Fewer than five density holes'});continue
        parts=split_data(frame.assign(target=0.),validation,target_column='target')
        save_split(parts,directory/'gamma_split.csv')
        train,val,test=parts
        # Require enough PRIMARY samples before expensive nested N fits.
        viable=[kind for kind in ('wet','dry') if all(len(p.loc[p.n_measured.notna() & p['gamma_'+kind].notna()])>=2 for p in parts)]
        if not viable:
            reports.append({'mode':mode,'status':'insufficient_data','reason':'Too few primary samples in one or more splits'});continue
        heldout=pd.concat([val,test],ignore_index=True)
        upstream=exclude_queries(spt,heldout,validation,options['xy_tolerance_m'])
        upstream[['boring_id','measurement_no','x','y','depth']].to_csv(directory/'allowed_spt_rows.csv',index=False)
        if upstream.boring_id.nunique()<5:
            reports.append({'mode':mode,'status':'insufficient_data','reason':'Too few SPT holes after exclusion'});continue
        print(f'N comparison {mode}: nested SPT fits; gamma train/validation/test={list(map(len,parts))}',flush=True)
        # Predict validation/test without using either area's/holes' SPT labels.
        predicted_holdout=fit_predict(upstream,heldout,validation,options,numeric,categorical,
                                     options['xgb_parameters'],directory/'n_models/heldout')
        val=predicted_holdout.loc[predicted_holdout.record_id.isin(val.record_id)].copy()
        test=predicted_holdout.loc[predicted_holdout.record_id.isin(test.record_id)].copy()
        # Cross-fitting supplies every gamma training row with an out-of-hole/area N.
        groups=spatial_keys(train,validation['block_size_m']) if mode=='spatial' else train.boring_id
        folds=min(options['cross_fit_folds'],groups.nunique())
        if folds<2:
            reports.append({'mode':mode,'status':'insufficient_data','reason':'Too few training groups for cross-fitting'});continue
        predictions=[]
        for fold,(_,query_index) in enumerate(GroupKFold(n_splits=folds).split(train,groups=groups)):
            query=train.iloc[query_index].copy()
            source=exclude_queries(upstream,query,validation,options['xy_tolerance_m'])
            predictions.append(fit_predict(source,query,validation,options,numeric,categorical,
                               options['xgb_parameters'],directory/f'n_models/train_fold_{fold}'))
        train=pd.concat(predictions,ignore_index=True)
        for name,part in zip(('train','validation','test'),(train,val,test)):
            part.to_csv(directory/f'{name}_n_features.csv',index=False)
        for kind in ('wet','dry'):
            try:
                rows=comparison_for_kind((train,val,test),kind,config,directory/kind)
            except InsufficientComparisonData as error:
                reports.append({'mode':mode,'kind':kind,'status':'insufficient_data','reason':str(error)})
                continue
            all_rows.extend(dict(mode=mode,kind=kind,**r) for r in rows)
            reports.append({'mode':mode,'kind':kind,'status':'complete'})
    pd.DataFrame(all_rows).to_csv(output/'comparison.csv',index=False)
    status='complete' if reports and all(r['status']=='complete' for r in reports) else 'incomplete'
    report={'status':status,'evaluations':reports,'distribution_model':config.get('distribution_model','no_n'), 'selection':'explicit config; no automatic selection by test score'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
