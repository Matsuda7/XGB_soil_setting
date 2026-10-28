"""Model A: effective cohesion c' from CU-bar/CD, using the N-model predictors."""
import json
from pathlib import Path
from module.output_layout import create_target_directory
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module.model_reporting import depth_scatter
from module.paths import RAW, TRAINING, project_path
from module.configuration import load_model_settings, load_prediction_settings
from module.model_reporting import save_model_reports
from module.xgb_common import load_model_config, build_preprocessor, build_regressor, regression_metrics
from module.gamma_features import load_dem, terrain_at, geographic_context, enrich, model_frame
from module.gamma_config import grid_shape
from module.validation import validate_config, split_data, save_split

SOURCE_REFERENCE='https://www.pref.yamaguchi.lg.jp/uploaded/attachment/189665.pdf'


def load_config():
    config=load_prediction_settings('config/c/prediction.json')
    config['spatial']={key:config[key] for key in ('grid_input','dem_input','dem_cache','grid_crs','grid_spacing_m','property_mask','property_mask_spacing_m','property_mask_bounds')}
    config['spatial']['grid_input']=project_path(config['spatial']['grid_input'])
    config['spatial']['dem_input']=project_path(config['spatial']['dem_input'])
    config['spatial']['dem_cache']=project_path(config['spatial']['dem_cache'])
    config['spatial']['property_mask']=project_path(config['spatial']['property_mask'])
    config['model_settings']=load_model_settings()
    grid_shape(config['spatial'])
    if config['spatial']['grid_crs'] != 'EPSG:6675':
        raise ValueError('Model A currently requires Japan Plane Rectangular CS VII (EPSG:6675)')
    config['numeric_features'],config['categorical_features']=load_model_config(project_path(config['model_config']))
    allowed={'n_value','x','y','surface_z','depth','elevation','sample_elevation','jshis_avs30','jshis_arv','slope','curvature','dem_missing'}
    if set(config['numeric_features'])-allowed or set(config['categorical_features'])-{'symbol','ser','jshis_jcode','soil_inventory_ssg'}:
        raise ValueError('Model A currently supports geographic N-model features only; target-derived RBF is not implemented')
    config['validation']=validate_config(json.loads(project_path(config['validation_config']).read_text()))
    from module.n_comparison_config import load_options
    config['n_comparison'] = load_options(config['n_estimation_config'])
    if 'n_value' in config['numeric_features'] and (not config['n_comparison']['enabled'] or config['validation']['mode']=='row'):
        raise ValueError('Model A n_value requires enabled N estimation and borehole/spatial validation')
    from module.phi_training import XGB_PARAMETERS
    config['xgb_parameters']=dict(XGB_PARAMETERS,**config.get('xgb_parameters',{}))
    if not isinstance(config.get('write_text_matrix',True),bool):
        raise ValueError('write_text_matrix must be a boolean')
    depth=config['prediction_depth_m']
    if isinstance(depth,bool) or not isinstance(depth,(int,float)) or not np.isfinite(depth) or depth<0:
        raise ValueError('Model A depth must be finite and non-negative')
    ratio=config['display_spacing_m']/config['spatial']['grid_spacing_m']
    if not np.isfinite(ratio) or ratio<1 or not np.isclose(ratio,round(ratio)):
        raise ValueError('Display spacing must be a multiple of grid spacing')
    size=config['input_chunk_size']
    if isinstance(size,bool) or not isinstance(size,int) or size<1: raise ValueError('Invalid chunk size')
    return config


def select_targets(raw):
    """Standard soil-test-list shear-strength intercept is c', not shear stress tau."""
    frame=raw.copy()
    code=frame.test_condition_code.fillna('').astype(str).str.strip().str.upper()
    label=frame.test_type.fillna('').astype(str)
    eligible=code.isin(['B0523','B0524']) | (code.eq('') & label.isin(['CUb','CD']))
    frame['target']=pd.to_numeric(frame.c_effective,errors='coerce')
    standard=pd.to_numeric(frame.shear_strength_effective,errors='coerce')
    frame['target_source']=np.where(frame.target.notna(),'explicit_c_effective','soiltestlist_effective_strength_intercept')
    conflict=frame.target.notna() & standard.notna() & ~np.isclose(frame.target,standard,rtol=0,atol=1e-6,equal_nan=True)
    frame['target']=frame.target.fillna(standard)
    frame['depth']=pd.to_numeric(frame.depth_mid,errors='coerce')
    reason=pd.Series('',index=frame.index)
    reason.loc[~eligible]='not_CUb_or_CD'
    reason.loc[eligible & (~np.isfinite(frame.target) | (frame.target<0))]='missing_or_invalid_effective_cohesion'
    reason.loc[eligible & conflict]='conflicting_effective_cohesion_fields'
    reason.loc[eligible & reason.eq('') & (~np.isfinite(frame.depth) | (frame.depth<0))]='invalid_depth'
    frame['target_unit']='kPa'
    frame['record_id']=np.arange(len(frame))
    rejected=frame.loc[reason.ne('')].copy();rejected['exclusion_reason']=reason.loc[reason.ne('')]
    return frame.loc[reason.eq('')].copy(),rejected


def signature(config):
    # load_config resolves spatial paths for file access; store them as strings
    # because preparation_summary.json must be JSON serializable.
    spatial = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config['spatial'].items()
    }
    return {'schema':'model_A_effective_cohesion_predicted_n_v2','spatial':spatial,
            'n_estimation':config['n_comparison'],
            'numeric_features':config['numeric_features'],'categorical_features':config['categorical_features'],
            'training_depth':config['model_settings']}


def prepare(raw=None):
    from module.soil_tests import load_test_records, project_records
    from module.n_matching import location_groups
    config=load_config();output=TRAINING/'c/model_A';output.mkdir(parents=True,exist_ok=True)
    raw=load_test_records(RAW/'soiltest/strength') if raw is None else raw
    frame,rejected=select_targets(raw)
    exclusions=[rejected]
    if frame.empty:
        rejected.to_csv(output/'excluded_records.csv',index=False)
        raise ValueError('No CU-bar/CD effective-cohesion records; total-stress fallback is forbidden')
    # Confirm the standardized XML context before using its c-prime field mapping.
    from xml.etree import ElementTree as ET
    for filename in frame.xml_file.unique():
        paths=list((RAW/'soiltest/strength').rglob(filename))
        if len(paths)!=1: raise ValueError(f'Ambiguous XML: {filename}')
        payload=paths[0].read_bytes()
        try: text=payload.decode('shift_jis')
        except UnicodeDecodeError: text=payload.decode('utf-8-sig')
        if ET.fromstring(text).tag.rsplit('}',1)[-1]!='SOILTESTLIST':
            raise ValueError(f'Unverified effective-cohesion XML schema: {filename}')
    frame,bad=project_records(frame,RAW/'soiltest/strength',config['spatial']['grid_crs'])
    exclusions.append(bad.assign(exclusion_reason='invalid_coordinates'))
    duplicate=frame.duplicated(['xml_file','sample_id','sample_no','depth','target','test_type'])
    exclusions.append(frame.loc[duplicate].assign(exclusion_reason='duplicate'))
    frame=frame.loc[~duplicate].copy()
    if frame.empty:
        pd.concat(exclusions,ignore_index=True).to_csv(output/'excluded_records.csv',index=False)
        raise ValueError('No Model A records with usable coordinates')
    dem=load_dem(config)
    outside=terrain_at(frame,dem,config).surface_z.isna()
    exclusions.append(frame.loc[outside].assign(exclusion_reason='outside_dem'))
    frame=frame.loc[~outside].copy()
    limits=config['model_settings']; depth=frame.depth
    in_range=pd.Series(True,index=frame.index)
    if limits['training_min_depth_m'] is not None: in_range &= depth >= limits['training_min_depth_m']
    if limits['training_max_depth_m'] is not None: in_range &= depth <= limits['training_max_depth_m']
    exclusions.append(frame.loc[~in_range].assign(exclusion_reason='outside_training_depth_range'))
    frame=frame.loc[in_range].copy()
    pd.concat(exclusions,ignore_index=True).to_csv(output/'excluded_records.csv',index=False)
    if frame.empty: raise ValueError('No Model A observations within the configured DEM and training-depth range')
    geology,jshis=geographic_context(config,frame)
    frame=enrich(frame,config,dem,geology,jshis)
    frame['sample_elevation']=frame.surface_z-frame.depth
    if 'n_value' in config['numeric_features']:
        from module.n_matching import prepare_n_data
        frame = prepare_n_data(frame,config,output,dem,geology,jshis)
        # Populate separately within each split; never use an in-sample global N feature.
        frame['n_value'] = np.nan
    frame['source_boring_id']=frame.boring_id
    frame['boring_id']=location_groups(frame,.01)
    frame.to_csv(output/'model_dataset.csv',index=False)
    report={'status':'prepared' if frame.boring_id.nunique()>=5 else 'insufficient_data',
            'rows':len(frame),'holes':int(frame.boring_id.nunique()),
            'test_types':frame.test_type.value_counts().to_dict(),'zero_targets':int(frame.target.eq(0).sum()),
            'target_range_kpa':[float(frame.target.min()),float(frame.target.max())],
            'training_depth_limits':config['model_settings'],
            'target_definition':"effective cohesion c-prime; soiltestlist effective-stress shear intercept",
            'unit':'kPa = kN/m2','source_reference':SOURCE_REFERENCE,'signature':signature(config)}
    (output/'preparation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def evaluate(data,config,output,mode,scenario="predicted_n",parts=None):
    from module.gamma_training import fit_evaluation
    output.mkdir(parents=True,exist_ok=True)
    if parts is None:
        parts=split_data(data,dict(config['validation'],mode=mode),target_column='target')
    if scenario == 'measured_n':
        parts=tuple(part.assign(n_value=part.n_measured) for part in parts)
    elif scenario != 'predicted_n':
        raise ValueError(f'Unsupported c N comparison scenario: {scenario}')
    elif 'n_value' in config['numeric_features']:
        from module.n_feature_pipeline import evaluation_features
        spt = pd.read_csv(TRAINING/'c/model_A/n_spt.csv')
        parts = evaluation_features(parts,spt,config,output/'n_models',dict(config['validation'],mode=mode))
    if min(map(len,parts))<2: raise ValueError('Model A evaluation requires at least two rows in each split')
    numeric,categorical=config['numeric_features'],config['categorical_features']
    model,transform,predicted=fit_evaluation(parts,numeric,categorical,config['xgb_parameters'])
    save_model_reports(model,transform,numeric,categorical,parts[0].target,output,
                       "Effective cohesion c' (kPa)")
    predicted=np.maximum(predicted,0.)
    test=parts[2];table=test.copy();table['predicted_c_effective_kpa']=predicted
    table.to_csv(output/'test_predictions.csv',index=False);save_split(parts,output/'split.csv')
    metrics={'test':regression_metrics(test.target,predicted),'best_n_estimators':int(model.best_iteration)+1,
             'mode':mode,'scenario':scenario,'rows':{k:len(p) for k,p in zip(('train','validation','test'),parts)},
             'holes':{k:int(p.boring_id.nunique()) for k,p in zip(('train','validation','test'),parts)},
             'unit':'kPa','prediction_lower_bound':0.}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,5));depth_scatter(ax,test.target,predicted,test.get('depth'),s=20,alpha=.7)
    high=max(float(test.target.max()),float(predicted.max()),1.)
    ax.plot([0,high],[0,high],'k--',lw=1)
    ax.set(xlabel="Observed effective cohesion c' (kPa)",ylabel="Predicted effective cohesion c' (kPa)",title=f'Model A ({mode})')
    ax.text(.04,.96,f"R² = {metrics['test']['r2']:.4f}",transform=ax.transAxes,ha='left',va='top',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    fig.tight_layout();fig.savefig(output/'observed_vs_predicted.png',dpi=180);plt.close(fig)
    return metrics


def compare_n_conditions(data,config,output,mode):
    """Evaluate predicted-N on all c samples; compare measured-N only if splitable."""
    if 'n_value' not in config['numeric_features']:
        return evaluate(data,config,output,mode)
    output.mkdir(parents=True,exist_ok=True)
    common=data.loc[data.n_measured.notna()].copy()
    required_holes=5
    measured_available=common.boring_id.nunique()>=required_holes
    results={}
    if measured_available:
        parts=split_data(common,dict(config['validation'],mode=mode),target_column='target')
        measured_available=min(map(len,parts))>=2
    if measured_available:
        evaluation_data=common
        parts=tuple(part.copy() for part in parts)
        save_split(parts,output/'common_split.csv')
        scenarios=config['n_comparison']['scenarios']
    else:
        # Preserve all eligible c samples; predicted N is available through the
        # leakage-safe upstream pipeline even when observed N does not match.
        evaluation_data=data
        parts=split_data(evaluation_data,dict(config['validation'],mode=mode),target_column='target')
        if min(map(len,parts))<2:
            raise ValueError('c N comparison requires at least two samples per validation split')
        save_split(parts,output/'predicted_n_split.csv')
        scenarios=['predicted_n']
    for scenario in scenarios:
        results[scenario]=evaluate(evaluation_data,config,output/scenario,mode,scenario,parts)
    comparison=[]
    for scenario in config['n_comparison']['scenarios']:
        if scenario in results:
            result=results[scenario]
            comparison.append({'scenario':scenario,'status':'evaluated',**result['test'],
                              'training_rows':result['rows']['train'],
                              'validation_rows':result['rows']['validation'],
                              'test_rows':result['rows']['test']})
        else:
            comparison.append({'scenario':scenario,'status':'unavailable',
                               'reason':f'measured-N matches only {common.boring_id.nunique()} borehole(s); '
                                        f'at least {required_holes} required for borehole validation',
                               'matched_rows':len(common),'matched_boreholes':int(common.boring_id.nunique())})
    pd.DataFrame(comparison).to_csv(output/'comparison.csv',index=False)
    return results['predicted_n']


def distribute():
    config = load_config()
    run = create_target_directory('c')
    config['output_dir'] = str(run)
    from module.config_snapshot import write_config_snapshot
    write_config_snapshot(config, run, 'c')
    status={'status':'running','model':'A','run':run.name,'target':"effective cohesion c'",'unit':'kPa'}
    try:
        preparation=json.loads((TRAINING/'c/model_A/preparation_summary.json').read_text())
        if preparation['signature']!=signature(config): raise ValueError('Rerun Model A preparation for the current features/spatial settings')
        data=pd.read_csv(TRAINING/'c/model_A/model_dataset.csv')
        (run/'preparation_summary.json').write_text(json.dumps(preparation,indent=2)+'\n')
        mode=config['validation']['mode'];primary=compare_n_conditions(data,config,run/'evaluation'/mode,mode)
        n_artifact = None
        if 'n_value' in config['numeric_features']:
            from module.n_feature_pipeline import crossfit_features, fit_grid_artifact
            spt = pd.read_csv(TRAINING/'c/model_A/n_spt.csv')
            data = crossfit_features(data,spt,config,run/'n_models/refit')
            data.to_csv(run/'final_training_n_features.csv',index=False)
            n_artifact = fit_grid_artifact(spt,data,config,run/'n_models/grid')
        numeric,categorical=config['numeric_features'],config['categorical_features']
        transform=build_preprocessor(numeric,categorical)
        x=transform.fit_transform(model_frame(data,numeric,categorical))
        parameters=dict(config['xgb_parameters'],n_estimators=primary['best_n_estimators'])
        parameters.pop('early_stopping_rounds',None)
        model=build_regressor(parameters);model.fit(x,data.target,verbose=False)
        output=run/'model';output.mkdir()
        model.save_model(output/'model.json');joblib.dump(transform,output/'preprocessor.joblib')
        save_model_reports(model,transform,numeric,categorical,data.target,output,
                           "Effective cohesion c' (kPa)",population='all-data final fitting records')
        metrics={'final_fit_rows':len(data),'final_fit_holes':int(data.boring_id.nunique()),
                 'numeric_features':numeric,'categorical_features':categorical,'xgb_parameters':parameters,
                 'training_depth_range_m':[float(data.depth.min()),float(data.depth.max())],
                 'target_definition':preparation['target_definition'],'unit':'kPa','evaluation':primary}
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        pd.DataFrame({'feature':transform.get_feature_names_out(),'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(output/'feature_importance.csv',index=False)
        import shutil
        evaluation_output=run/'evaluation'/mode
        if 'n_value' in numeric: evaluation_output=evaluation_output/'predicted_n'
        shutil.copyfile(evaluation_output/'observed_vs_predicted.png',output/'observed_vs_predicted_test.png')
        shutil.copyfile(evaluation_output/'test_predictions.csv',output/'test_predictions.csv')
        from module.all_observed_reporting import save_all_observed
        metrics['all_observed'] = save_all_observed(
            data, np.maximum(model.predict(x), 0.), 'target', 'predicted_c_kpa', output,
            'observed_vs_predicted.png', "effective cohesion c' [kPa]")
        from module.c_prediction import predict_grid
        status.update(predict_grid(config,model,transform,metrics,run/'grid',n_artifact=n_artifact))
    except Exception as error:
        status.update(status='failed',error=str(error))
        (run/'run_summary.json').write_text(json.dumps(status,indent=2)+'\n')
        raise
    (run/'run_summary.json').write_text(json.dumps(status,indent=2)+'\n')
    return {**status,'output':str(run)}
