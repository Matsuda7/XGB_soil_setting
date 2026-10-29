"""Resumable 10 m distributions from an already evaluated comparison run.

Prepare the geographic feature cache and XGBoost grids first; TabPFN consumes
exactly the same cache. Completed batches are durable and never reinterpreted
as full-area predictions. No training or prediction rows are subsampled.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from module.paths import project_path
from module.model_comparison import sha256, environment, dense
from module.comparison_data import load_target, TARGETS
from module.comparison_models import build_model, save_model, predict_batches, initialize_tabpfn, validate_saved_checkpoint
from module.tabpfn_runtime import batch_size as configured_batch_size, runtime_override, append_log
from module.comparison_distribution import grid_chunks
from module.gamma_config import grid_shape
from module.gamma_features import model_frame
from module.gamma_prediction import grid_indices
from module.xgb_common import build_preprocessor


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


@contextmanager
def job_lock(path):
    """Refuse concurrent writers, including when resuming a long CPU job."""
    with path.open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another writer holds {path}') from error
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def initialize(source, output):
    source = source.resolve()
    if (output/'manifest.json').exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['comparison_source'] != str(source):
            raise ValueError('Output belongs to a different comparison run')
        return json.loads((output/'config.json').read_text()), json.loads((output/'spatial.json').read_text())
    config = json.loads((source/'config.json').read_text())
    spatial = json.loads(project_path('config/spatial.json').read_text())
    summary = pd.read_csv(source/'metrics_summary.csv')
    eligible = {}
    for target in config['targets']:
        eligible[target] = [m for m in config['models'] if ((summary.target==target)&(summary.model==m)).any()]
        for model in eligible[target]:
            oof = pd.read_csv(source/target/'oof_predictions.csv')
            rows = oof[oof.model==model]
            expected = json.loads((source/target/'dataset.json').read_text())['rows']
            if len(rows)!=expected or rows.row_id.nunique()!=expected:
                raise ValueError(f'{target}/{model}: incomplete OOF')
    if not all('xgboost' in v for v in eligible.values()):
        raise ValueError('This runner requires completed XGBoost CV for every target')
    config['distribution'] = dict(enabled=True, depth_m=5., chunk_size=100000, feature_csv=None)
    config['eligible_models'] = eligible
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'config.json', config)
    atomic_json(output/'spatial.json', spatial)
    metadata = dict(comparison_source=str(source), created_at=now(), environment=environment(),
                    resolution_m=spatial['grid_spacing_m'], depth_m=5.,
                    shape=grid_shape(spatial), eligible_models=eligible,
                    excluded_models={f'{t}/{m}':'No completed CV result; consult source failure.json'
                                     for t in config['targets'] for m in config['models'] if m not in eligible[t]},
                    inputs={k:sha256(project_path(spatial[k])) for k in ('grid_input','dem_input','property_mask')},
                    cv_config_sha256=sha256(source/'config.json'),
                    matrix_orientation='row index increases northward, column index eastward; NaN outside mask',
                    prediction_policy='raw predictions; N-to-phi uses existing clipped conversion only')
    atomic_json(output/'manifest.json', metadata)
    return config, spatial


def fit_or_load(target, name, config, source, output, cache_mode='low_memory'):
    spec = config['targets'][target]
    expected = json.loads((source/target/'dataset.json').read_text())['source_sha256']
    if sha256(project_path(spec['data'])) != expected:
        raise ValueError(f'{target}: training CSV changed since evaluation')
    dest = output/target/name
    dest.mkdir(parents=True, exist_ok=True)
    model_file = dest/('model.json' if name=='xgboost' else 'model.tabpfn_fit')
    if model_file.exists() and (dest/'fitted.json').exists():
        pre = joblib.load(dest/'preprocessor.joblib')
        if name=='xgboost':
            from xgboost import XGBRegressor
            model = XGBRegressor(n_jobs=4); model.load_model(model_file)
        else:
            runtime = initialize_tabpfn(config['tabpfn'])
            validate_saved_checkpoint(model_file, config['tabpfn'])
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor.load_from_fit_state(str(model_file), device=runtime['device'])
            model._soil_runtime = runtime
            append_log(dest / 'runtime.jsonl', runtime)
        return model, pre
    data,_,y,unit = load_target(target,spec,config)
    pre = build_preprocessor(spec['numeric_features'],spec['categorical_features'])
    x = dense(pre.fit_transform(model_frame(data,spec['numeric_features'],spec['categorical_features'])))
    model = build_model(name,config,len(x),x.shape[1])
    if name=='tabpfn':
        model.set_params(fit_mode=cache_mode)
    started=time.perf_counter();model.fit(x,data[y].to_numpy())
    joblib.dump(pre,dest/'preprocessor.joblib');save_model(model,name,dest)
    atomic_json(dest/'fitted.json',dict(rows=len(data),unit=unit,parameters=model.get_params(),
                fitted_at=now(),fit_seconds=time.perf_counter()-started,source_sha256=expected))
    return model,pre


def matrix(path,shape):
    if path.exists():
        result=np.load(path,mmap_mode='r+')
        if result.shape != shape: raise ValueError('Existing matrix shape mismatch')
        return result
    result=np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=np.float32)
    result[:]=np.nan;result.flush()
    return result


def plot_matrix(path,spatial,title,unit,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid=np.load(path,mmap_mode='r')
    b=spatial['property_mask_bounds'];step=spatial['grid_spacing_m'];half=step/2
    fig,ax=plt.subplots(figsize=(8,8))
    im=ax.imshow(grid,origin='lower',extent=[b['xmin']-half,b['xmax']+half,b['ymin']-half,b['ymax']+half],
                 interpolation='nearest',cmap='viridis')
    ax.set(title=title+'\nDepth 5 m; prediction mesh 10 m',xlabel='X [m] (EPSG:6675)',ylabel='Y [m]',aspect='equal')
    fig.colorbar(im,ax=ax,label=unit,shrink=.8)
    fig.tight_layout();fig.savefig(destination,dpi=170);plt.close(fig)


def prepare(config,spatial,source,output):
    status_path=output/'prepare_status.json'
    if status_path.exists() and json.loads(status_path.read_text())['status']=='complete': return
    shape=grid_shape(spatial)
    cache=output/'features';cache.mkdir(exist_ok=True)
    fitted={t:fit_or_load(t,'xgboost',config,source,output) for t in config['targets']}
    grids={t:matrix(output/t/'xgboost/raw.npy',shape) for t in fitted}
    phi=matrix(output/'N/xgboost/phi_deg.npy',shape) if 'N' in fitted else None
    processed=0;chunks=[];started=time.perf_counter()
    for i,frame in enumerate(grid_chunks(config),1):
        feature_path=cache/f'{i:06d}.csv.gz'
        marker=cache/f'{i:06d}.json'
        if marker.exists():
            item=json.loads(marker.read_text());processed+=item['rows'];chunks.append(item);continue
        xi,yi,inside=grid_indices(frame,spatial,shape)
        if not inside.all():raise ValueError('Outside prediction coordinates')
        for target,(model,pre) in fitted.items():
            spec=config['targets'][target]
            x=dense(pre.transform(model_frame(frame,spec['numeric_features'],spec['categorical_features'])))
            values=model.predict(x)
            if not np.isfinite(values).all():raise ValueError('Nonfinite XGBoost prediction')
            grids[target][yi,xi]=values;grids[target].flush()
            if target=='N':
                from module.phi_prediction import PHI_MAX_DEG
                phi[yi,xi]=np.minimum(np.sqrt(15*np.clip(values,0,50))+15,PHI_MAX_DEG);phi.flush()
        cols=list(dict.fromkeys(['x','y','depth']+[f for s in config['targets'].values() for f in s['numeric_features']+s['categorical_features']]))
        temp=feature_path.with_suffix('.tmp')
        frame[cols].to_csv(temp,index=False,compression='gzip');temp.replace(feature_path)
        item=dict(chunk=i,rows=len(frame),file=feature_path.name,sha256=sha256(feature_path))
        atomic_json(marker,item);chunks.append(item);processed+=len(frame)
        status=dict(status='running',rows=processed,chunks=len(chunks),updated_at=now(),elapsed_seconds=time.perf_counter()-started)
        atomic_json(status_path,status);print('XGBoost/grid',processed,'rows',flush=True)
    # The shared iterator verifies complete coverage of the property mask.
    atomic_json(output/'feature_manifest.json',dict(rows=processed,chunks=chunks))
    for target in fitted:
        plot_matrix(output/target/'xgboost/raw.npy',spatial,f'{target}: XGBoost',TARGETS[target][1],output/target/'xgboost/distribution.png')
    if phi is not None:
        plot_matrix(output/'N/xgboost/phi_deg.npy',spatial,'Converted phi: XGBoost','degrees',output/'N/xgboost/phi_distribution.png')
    atomic_json(status_path,dict(status='complete',rows=processed,chunks=len(chunks),completed_at=now()))


def recompute_xgboost(config,spatial,source,output):
    """Reuse the verified feature cache, with the same dense input used in CV."""
    manifest=json.loads((output/'feature_manifest.json').read_text())
    shape=grid_shape(spatial)
    fitted={t:fit_or_load(t,'xgboost',config,source,output) for t in config['targets']}
    grids={t:matrix(output/t/'xgboost/raw.npy',shape) for t in fitted}
    phi=matrix(output/'N/xgboost/phi_deg.npy',shape) if 'N' in fitted else None
    total=0
    for item in manifest['chunks']:
        path=output/'features'/item['file']
        if sha256(path)!=item['sha256']:raise ValueError('Feature cache changed')
        frame=pd.read_csv(path);xi,yi,inside=grid_indices(frame,spatial,shape)
        if not inside.all():raise ValueError('Outside prediction coordinates')
        for target,(model,pre) in fitted.items():
            spec=config['targets'][target]
            x=dense(pre.transform(model_frame(frame,spec['numeric_features'],spec['categorical_features'])))
            values=model.predict(x)
            if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
            grids[target][yi,xi]=values;grids[target].flush()
            if target=='N':
                from module.phi_prediction import PHI_MAX_DEG
                phi[yi,xi]=np.minimum(np.sqrt(15*np.clip(values,0,50))+15,PHI_MAX_DEG);phi.flush()
        total+=len(frame)
        atomic_json(output/'prepare_status.json',dict(status='running',stage='dense_xgboost',rows=total,total_rows=manifest['rows'],updated_at=now()))
        print('XGBoost dense',total,'/',manifest['rows'],flush=True)
    if total!=manifest['rows']:raise ValueError('Prediction row count mismatch')
    for t in fitted:
        plot_matrix(output/t/'xgboost/raw.npy',spatial,f'{t}: XGBoost (raw)',TARGETS[t][1],output/t/'xgboost/distribution.png')
    if phi is not None:plot_matrix(output/'N/xgboost/phi_deg.npy',spatial,'Converted phi: XGBoost','degrees',output/'N/xgboost/phi_distribution.png')
    atomic_json(output/'prepare_status.json',dict(status='complete',stage='dense_xgboost',rows=total,chunks=len(manifest['chunks']),completed_at=now()))


def tabpfn(config,spatial,source,output,cache_mode,batch_size,target=None):
    if cache_mode != config['tabpfn'].get('fit_mode','low_memory'):
        raise ValueError('Inference mode must match the evaluated CV protocol')
    manifest=json.loads((output/'feature_manifest.json').read_text())
    shape=grid_shape(spatial)
    targets=[t for t,v in config['eligible_models'].items() if 'tabpfn' in v and (target is None or target==t)]
    for t in targets:
        dest=output/t/'tabpfn';status_path=dest/'prediction_status.json'
        if status_path.exists() and json.loads(status_path.read_text())['status']=='complete':continue
        model,pre=fit_or_load(t,'tabpfn',config,source,output,cache_mode)
        append_log(dest/'runtime.jsonl', dict(**model._soil_runtime, batch_size=batch_size,
                                           prediction_points=manifest['rows'], event='start_or_resume'))
        grid=matrix(dest/'raw.npy',shape)
        state=json.loads(status_path.read_text()) if status_path.exists() else dict(chunk=0,offset=0,rows=0)
        total=state['rows'];started=time.perf_counter();initial=total
        for chunk_index,item in enumerate(manifest['chunks']):
            if chunk_index < state['chunk']:continue
            path=output/'features'/item['file']
            if sha256(path)!=item['sha256']:raise ValueError('Feature cache changed')
            frame=pd.read_csv(path);spec=config['targets'][t]
            x=dense(pre.transform(model_frame(frame,spec['numeric_features'],spec['categorical_features'])))
            xi,yi,_=grid_indices(frame,spatial,shape)
            offset=state['offset'] if chunk_index==state['chunk'] else 0
            for start in range(offset,len(frame),batch_size):
                if (output/'STOP_TABPFN').exists():
                    atomic_json(status_path,dict(status='stopped',chunk=chunk_index,offset=start,rows=total,
                                total_rows=manifest['rows'],updated_at=now()))
                    print('Stopped at a saved batch boundary; remove STOP_TABPFN to resume.',flush=True)
                    return
                stop=min(start+batch_size,len(frame))
                values=predict_batches(model,x[start:stop],batch_size,dest/'prediction_runtime.jsonl')
                if not np.isfinite(values).all():raise ValueError('Nonfinite TabPFN prediction')
                grid[yi[start:stop],xi[start:stop]]=values;grid.flush();total+=stop-start
                elapsed=time.perf_counter()-started
                atomic_json(status_path,dict(status='running',chunk=chunk_index,offset=stop,rows=total,total_rows=manifest['rows'],
                            updated_at=now(),elapsed_seconds=elapsed,rows_per_second=(total-initial)/max(elapsed,1e-6)))
                print(t,'TabPFN',total,'/',manifest['rows'],flush=True)
        if total!=manifest['rows']:raise ValueError('TabPFN coverage count mismatch')
        plot_matrix(dest/'raw.npy',spatial,f'{t}: TabPFN',TARGETS[t][1],dest/'distribution.png')
        xgb=np.load(output/t/'xgboost/raw.npy',mmap_mode='r')
        delta=matrix(output/t/'difference.npy',shape)
        for row in range(0,shape[0],100):delta[row:row+100]=grid[row:row+100]-xgb[row:row+100]
        delta.flush()
        plot_matrix(output/t/'difference.npy',spatial,f'{t}: TabPFN - XGBoost',TARGETS[t][1],output/t/'difference.png')
        atomic_json(status_path,dict(status='complete',rows=total,total_rows=manifest['rows'],completed_at=now()))
        del model


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['prepare','xgboost','tabpfn','all'],default='all')
    parser.add_argument('--cache-mode',choices=['low_memory','fit_preprocessors','fit_with_cache'],default='low_memory')
    parser.add_argument('--batch-size',type=int,default=None,help='Override tabpfn.predict_batch_size')
    parser.add_argument('--runtime-config',help='JSON with tabpfn CPU threads/device/batch size overrides only')
    parser.add_argument('--target',choices=['c','gamma_wet','gamma_dry'])
    parser.add_argument('--plan',action='store_true',help='Show requested actions without fitting, predicting or writing results')
    parser.add_argument('--resume',action='store_true',help='Explicitly clear STOP_TABPFN and resume saved batches')
    args=parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size<1:parser.error('batch-size must be positive')
    if args.resume and args.stage not in ('tabpfn','all'):
        parser.error('--resume requires --stage tabpfn or all')
    source=project_path(args.comparison_run);output=project_path(args.output)
    if args.plan:
        print(json.dumps(dict(comparison_run=str(source),output=str(output),stage=args.stage,
                         target=args.target,batch_size=args.batch_size,cache_mode=args.cache_mode,
                         resume=args.resume,analysis_executed=False),indent=2))
        return 0
    if args.stage in ('tabpfn','all') and (output/'STOP_TABPFN').exists() and not args.resume:
        parser.error('STOP_TABPFN is present. Use --resume only when you want to restart analysis.')
    config,spatial=initialize(source,output)
    if args.stage in ('tabpfn','all'):
        runtime_override(config,args.runtime_config)
        args.batch_size = args.batch_size if args.batch_size is not None else configured_batch_size('tabpfn',config)
    with ExitStack() as stack:
        if args.stage in ('prepare','xgboost','all'):stack.enter_context(job_lock(output/'xgboost.lock'))
        if args.stage in ('tabpfn','all'):stack.enter_context(job_lock(output/'tabpfn.lock'))
        if args.resume:(output/'STOP_TABPFN').unlink(missing_ok=True)
        if args.stage in ('prepare','all'):prepare(config,spatial,source,output)
        if args.stage=='xgboost':recompute_xgboost(config,spatial,source,output)
        if args.stage in ('tabpfn','all'):tabpfn(config,spatial,source,output,args.cache_mode,args.batch_size,args.target)
    return 0


if __name__=='__main__':raise SystemExit(main())
