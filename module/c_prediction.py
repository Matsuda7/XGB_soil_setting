"""Stream Model A effective-cohesion predictions on the shared phi grid."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from module.paths import project_path
from module.gamma_config import grid_shape
from module.gamma_prediction import grid_indices
from module.gamma_features import load_dem, geographic_context, enrich, model_frame
from module.raster import read_binary_mask, boundary_segments, save_text_matrix


def predict_grid(config,model,transform,metrics,output,n_artifact=None):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('Model A grid output must be empty')
    chunks=output/'chunks';chunks.mkdir()
    spatial=config['spatial'];shape=grid_shape(spatial)
    mask=read_binary_mask(project_path(spatial['property_mask']),shape)
    grid=np.lib.format.open_memmap(output/'c_effective_kpa.npy',mode='w+',dtype=np.float32,shape=shape)
    grid[:]=np.nan;seen=np.zeros(shape,dtype=bool)
    dem=load_dem(config);geology,jshis=geographic_context(config)
    total=0;outside=0;zero=0;negative=0
    reader=pd.read_csv(project_path(spatial['grid_input']),sep=r'\s+',header=None,
                       names=['x','y','legacy_z','legacy_bedrock'],usecols=[0,1],chunksize=config['input_chunk_size'])
    for number,points in enumerate(reader,start=1):
        points=points.apply(pd.to_numeric,errors='coerce')
        xi,yi,inside=grid_indices(points,spatial,shape)
        active=np.zeros(len(points),dtype=bool);active[inside]=mask[yi[inside],xi[inside]]!=0
        outside+=int((~inside).sum());zero+=int((inside & ~active).sum())
        points=points.loc[active].reset_index(drop=True);xi=xi[active];yi=yi[active]
        if points.empty: continue
        flat=yi*shape[1]+xi
        if len(np.unique(flat))!=len(flat) or seen[yi,xi].any(): raise ValueError('Duplicate active grid coordinates')
        points['depth']=config['prediction_depth_m']
        frame=enrich(points,config,dem,geology,jshis)
        frame['n_value_elevation']=frame.surface_z-frame.depth
        if 'n_input' in config['numeric_features']:
            if n_artifact is None: raise ValueError('Missing upstream N model for c prediction')
            from module.gamma_predicted_n import add_predicted_n
            frame = add_predicted_n(frame,n_artifact)
        values=model.predict(transform.transform(model_frame(frame,config['numeric_features'],config['categorical_features'])))
        if not np.isfinite(values).all(): raise ValueError('Non-finite Model A predictions')
        negative+=int((values<0).sum());values=np.maximum(values,0.)
        grid[yi,xi]=values;seen[yi,xi]=True
        table=points[['x','y','depth']].copy();table['c_effective_kpa']=values
        table['sample_z']=frame.sample_z
        if n_artifact is not None:
            for name in ('n_input','n_ensemble_std','n_lower','n_upper'): table[name]=frame[name]
        path=chunks/f'c_effective_{number:06d}.csv.gz';temporary=path.with_suffix('.gz.tmp')
        table.to_csv(temporary,index=False,compression={'method':'gzip','compresslevel':1});temporary.replace(path)
        total+=len(points)
        print(f'Model A chunk {number}: total={total:,}',flush=True)
    grid.flush();missing=int(np.count_nonzero((mask!=0)&~seen))
    report={'status':'complete' if total>0 and missing==0 else 'incomplete','model':'A',
            'prediction_depth_m':config['prediction_depth_m'],'predicted_points':total,
            'missing_active_cells':missing,'outside_grid_rows':outside,'prop_zero_rows_skipped':zero,
            'negative_predictions_clipped_to_zero':negative,'unit':'kPa','shape':list(shape),
            'crs':spatial['grid_crs'],'bounds':spatial['property_mask_bounds'],
            'grid_spacing_m':spatial['grid_spacing_m'],'display_spacing_m':config['display_spacing_m'],
            'matrix_orientation':'rows northward; columns eastward','nodata':'NaN',
            'depth_outside_training_range':not(metrics['training_depth_range_m'][0]<=config['prediction_depth_m']<=metrics['training_depth_range_m'][1])}
    if report['status']=='complete':
        if config.get('write_text_matrix',True):
            save_text_matrix(output/'c_effective_kpa.txt.gz',grid,mask)
        stride=int(round(config['display_spacing_m']/spatial['grid_spacing_m']))
        sampled=np.array(grid[::stride,::stride]);bounds=spatial['property_mask_bounds'];step=config['display_spacing_m'];half=step/2
        if not np.isfinite(sampled).any(): raise ValueError('No populated display cells; reduce display_spacing_m')
        extent=[bounds['xmin']-half,bounds['xmin']+(sampled.shape[1]-1)*step+half,
                bounds['ymin']-half,bounds['ymin']+(sampled.shape[0]-1)*step+half]
        fig,ax=plt.subplots(figsize=(8,8))
        image=ax.imshow(sampled,origin='lower',extent=extent,cmap='viridis',interpolation='nearest')
        ax.add_collection(LineCollection(boundary_segments(mask,bounds,spatial['grid_spacing_m']),colors='#333333',linewidths=.6))
        fig.colorbar(image,ax=ax,label="Effective cohesion c' (kPa)",shrink=.8)
        ax.set(title=f"Model A: Effective cohesion at depth {config['prediction_depth_m']:g} m",
               xlabel='Y (m) — Japan Plane Rectangular CS VII',ylabel='X (m) — Japan Plane Rectangular CS VII',aspect='equal')
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
        fig.tight_layout();fig.savefig(output/'c_effective.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (output/'prediction_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
