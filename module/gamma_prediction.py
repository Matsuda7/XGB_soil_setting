"""Chunked wet/dry inference and prop-masked gamma distribution products."""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from module.paths import project_path
from module.gamma_config import KINDS, grid_shape
from module.gamma_features import load_dem, geographic_context, enrich, model_frame
from module.raster import read_binary_mask,boundary_segments,save_text_matrix
LOGGER=logging.getLogger(__name__)


def grid_indices(points,spatial,shape):
    b=spatial['property_mask_bounds'];step=float(spatial['grid_spacing_m'])
    x=points.x.to_numpy(dtype=float);y=points.y.to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite prediction coordinates')
    xi=np.rint((x-b['xmin'])/step).astype(np.int64);yi=np.rint((y-b['ymin'])/step).astype(np.int64)
    inside=(xi>=0)&(xi<shape[1])&(yi>=0)&(yi<shape[0])
    aligned=np.isclose(x,b['xmin']+xi*step,rtol=0,atol=1e-4)&np.isclose(y,b['ymin']+yi*step,rtol=0,atol=1e-4)
    if np.any(inside & ~aligned):
        raise ValueError('Prediction grid is not aligned with configured prop/DEM spacing')
    return xi,yi,inside


def predict_points(points,depth,config,context,models,n_artifact=None):
    frame=points[['x','y']].copy();frame['depth']=float(depth)
    dem,geology,jshis=context
    enriched=enrich(frame,config,dem,geology,jshis)
    if config.get('distribution_model', 'no_n') == 'predicted_n':
        if n_artifact is None:
            raise ValueError('Missing N model for predicted_n distribution')
        from module.gamma_predicted_n import add_predicted_n
        enriched = add_predicted_n(enriched, n_artifact)
    output=enriched[['x','y','sample_z']].copy()
    if n_artifact is not None:
        for name in ('n_input','n_ensemble_std','n_lower','n_upper'):
            output[name] = enriched[name]
    for kind,(model,preprocessor,metrics) in models.items():
        features=model_frame(enriched,metrics['numeric_features'],metrics['categorical_features'])
        values=model.predict(preprocessor.transform(features))
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite {kind} gamma prediction')
        output[f'gamma_{kind}_kn_m3']=values
        output[f'density_{kind}_g_cm3']=values/config['gravity_m_s2']
    return output


def plot_maps(matrices,mask,config,output):
    spatial=config['spatial'];bounds=spatial['property_mask_bounds'];step=spatial['grid_spacing_m']
    stride=int(round(config['display_spacing_m']/step))
    segments=boundary_segments(mask,bounds,step)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    for kind,grid in matrices.items():
        sampled=np.array(grid[::stride,::stride])
        if not np.isfinite(sampled).any():
            raise ValueError('No populated display cells; reduce display_spacing_m')
        # Sampling is only for the PNG. Numerical matrices retain the configured full grid.
        fig,ax=plt.subplots(figsize=(8,8))
        half=config['display_spacing_m']/2
        extent=[(bounds['xmin']-half),(bounds['xmin']+(sampled.shape[1]-1)*config['display_spacing_m']+half),
                (bounds['ymin']-half),(bounds['ymin']+(sampled.shape[0]-1)*config['display_spacing_m']+half)]
        image=ax.imshow(sampled,origin='lower',extent=extent,cmap='viridis',interpolation='nearest')
        ax.add_collection(LineCollection(segments,colors='#333333',linewidths=.6))
        fig.colorbar(image,ax=ax,label=f'γ_{kind} [kN/m³]',shrink=.8)
        label='Wet' if kind=='wet' else 'Dry'
        ax.set(title=f'{label} unit weight at depth {config["prediction_depth_m"]:g} m\nModel: {config.get("distribution_model", "no_n")}',
               xlabel='Y (m) — Japan Plane Rectangular CS VII',ylabel='X (m) — Japan Plane Rectangular CS VII',aspect='equal')
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
        fig.tight_layout();fig.savefig(Path(output)/f'gamma_{kind}.png',dpi=180,bbox_inches='tight');plt.close(fig)


def predict_grid(config,models,output,max_chunks=None,n_artifact=None):
    """A fresh output directory isolates models/depths; no implicit resume."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    chunks_dir=output/'chunks'
    if any(output.iterdir()):
        raise FileExistsError(f'Gamma prediction needs a new directory: {output}')
    chunks_dir.mkdir()
    spatial=config['spatial'];shape=grid_shape(spatial)
    mask=read_binary_mask(project_path(spatial['property_mask']),shape)
    context=(load_dem(config),*geographic_context(config))
    matrices={kind:np.lib.format.open_memmap(output/f'gamma_{kind}.npy',mode='w+',dtype=np.float32,shape=shape) for kind in KINDS}
    for grid in matrices.values():grid[:]=np.nan
    seen=np.zeros(shape,dtype=bool)
    total=0;processed=0;outside=0;skipped=0;violations=0;partial=False
    csv=pd.read_csv(project_path(spatial['grid_input']),sep=r'\s+',header=None,names=['x','y','legacy_surface_z','legacy_bedrock_z'],usecols=[0,1],chunksize=config['input_chunk_size'])
    for number,points in enumerate(csv,start=1):
        if max_chunks is not None and number>max_chunks:
            partial=True;break
        points=points.apply(pd.to_numeric,errors='coerce')
        xi,yi,inside=grid_indices(points,spatial,shape)
        outside+=int((~inside).sum())
        active=np.zeros(len(points),dtype=bool)
        active[inside]=mask[yi[inside],xi[inside]]!=0
        skipped+=int((inside & ~active).sum())
        points=points.loc[active].reset_index(drop=True);xi=xi[active];yi=yi[active]
        processed+=1
        if points.empty:
            continue
        flat=yi*shape[1]+xi
        if len(np.unique(flat))!=len(flat) or seen[yi,xi].any():
            raise ValueError(f'Duplicate active grid coordinates in chunk {number}')
        predicted=predict_points(points,config['prediction_depth_m'],config,context,models,n_artifact)
        violations+=int((predicted.gamma_dry_kn_m3>predicted.gamma_wet_kn_m3).sum())
        for kind,grid in matrices.items():grid[yi,xi]=predicted[f'gamma_{kind}_kn_m3'].to_numpy(dtype=np.float32)
        seen[yi,xi]=True
        path=chunks_dir/f'gamma_{number:06d}.csv.gz';temporary=path.with_suffix('.gz.tmp')
        predicted.to_csv(temporary,index=False,compression={'method':'gzip','compresslevel':1})
        temporary.replace(path)
        total+=len(predicted)
        print(f'gamma chunk {number}: {len(predicted):,} points; total={total:,}',flush=True)
    missing=int(np.count_nonzero((mask!=0)&~seen))
    for grid in matrices.values():grid.flush()
    status='complete' if not partial and missing==0 and total>0 else 'incomplete'
    report={'status':status,'distribution_model':config.get('distribution_model','no_n'),'prediction_depth_m':config['prediction_depth_m'],'predicted_points':total,
        'missing_active_cells':missing,'outside_grid_rows':outside,'prop_zero_rows_skipped':skipped,
        'processed_chunks':processed,'dry_above_wet_predictions':violations,
        'shape':list(shape),'grid_spacing_m':spatial['grid_spacing_m'],'display_spacing_m':config['display_spacing_m'],
        'crs':spatial['grid_crs'],'bounds':spatial['property_mask_bounds'],'nodata':'NaN (prop=0 or unpredicted)',
        'target_unit':'kN/m3','matrix_orientation':'rows increase northward; columns increase eastward',
        'depth_outside_training_range':{k: not (m[2]['training_depth_range_m'][0]<=config['prediction_depth_m']<=m[2]['training_depth_range_m'][1]) for k,m in models.items()}}
    (output/'prediction_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # Partial computations retain diagnostic arrays/chunks, but do not publish final maps.
    if status=='complete':
        if config['write_text_matrix']:
            for kind,grid in matrices.items():
                save_text_matrix(output/f'gamma_{kind}.txt.gz',grid,mask)
        plot_maps(matrices,mask,config,output)
    return report
