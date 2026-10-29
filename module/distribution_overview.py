"""Audit completed 10 m grids and publish explicitly bounded display maps."""
import argparse
import json
from pathlib import Path
import numpy as np
from module.paths import project_path
from module.full_distribution import atomic_json, plot_matrix, now
from module.gamma_config import grid_shape
from module.raster import read_binary_mask


def publish(root):
    state=json.loads((root/'prepare_status.json').read_text())
    if state['status']!='complete':raise ValueError('XGBoost full-area calculation is not complete')
    spatial=json.loads((root/'spatial.json').read_text());shape=grid_shape(spatial)
    mask=read_binary_mask(project_path(spatial['property_mask']),shape)!=0
    expected=int(mask.sum());audit={}
    for target in ['c','N','gamma_wet','gamma_dry']:
        path=root/target/'xgboost/raw.npy';raw=np.load(path,mmap_mode='r')
        values=raw[mask]
        if not np.isfinite(values).all() or np.isfinite(raw[~mask]).any():
            raise ValueError(f'{target}: incomplete coverage or values outside mask')
        audit[target]=dict(cells=expected,min=float(values.min()),max=float(values.max()),negative_cells=int((values<0).sum()))
        if target=='N':audit[target]['above_target_cap_cells']=int((values>50).sum())
        if target in ('c','N'):
            bounded=np.lib.format.open_memmap(root/target/'xgboost/bounded.npy',mode='w+',shape=shape,dtype=np.float32)
            for row in range(0,shape[0],100):
                block=raw[row:row+100]
                bounded[row:row+100]=np.maximum(block,0) if target=='c' else np.clip(block,0,50)
            bounded.flush()
            label="c' = max(raw c', 0)" if target=='c' else 'N = clip(raw N, 0, 50)'
            plot_matrix(root/target/'xgboost/bounded.npy',spatial,'XGBoost: '+label,'kPa' if target=='c' else 'N',root/target/'xgboost/bounded_distribution.png')
    wet=np.load(root/'gamma_wet/xgboost/raw.npy',mmap_mode='r')
    dry=np.load(root/'gamma_dry/xgboost/raw.npy',mmap_mode='r')
    audit['gamma_dry_above_wet_cells']=int(((dry>wet)&mask).sum())
    audit['corrections']={'c':'max(raw,0) for bounded map only','N':'clip(raw,0,50) for bounded N and phi conversion only',
                          'gamma':'no correction','raw_files':'unchanged; raw.npy and distribution.png'}
    audit['verified_at']=now();atomic_json(root/'xgboost_full_grid_audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b=spatial['property_mask_bounds'];h=spatial['grid_spacing_m']/2
    extent=[b['xmin']-h,b['xmax']+h,b['ymin']-h,b['ymax']+h]
    fig,axes=plt.subplots(2,3,figsize=(17,11))
    items=[('c','bounded.npy',"Effective cohesion c' (negative values set to 0)",'kPa'),
           ('N','bounded.npy','SPT N-value (clipped to target range 0-50)','N'),
           ('N','phi_deg.npy','Converted phi (not measured phi)','degrees'),
           ('gamma_wet','raw.npy','Wet unit weight (raw)','kN/m3'),
           ('gamma_dry','raw.npy','Dry unit weight (raw)','kN/m3')]
    for ax,(target,name,label,unit) in zip(axes.ravel(),items):
        grid=np.load(root/target/'xgboost'/name,mmap_mode='r')
        im=ax.imshow(grid,origin='lower',extent=extent,cmap='viridis',interpolation='nearest')
        ax.set(title=label,xlabel='X [m]',ylabel='Y [m]',aspect='equal')
        fig.colorbar(im,ax=ax,label=unit,shrink=.76)
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(.05,.85,f'XGBoost: full area completed\n\nDepth: 5 m\nPrediction mesh: 10 m\nActive cells: {expected:,}\nCRS: EPSG:6675\n\nRaw predictions are retained separately.\nTabPFN full-area maps are incomplete.\nNo TabPFN values are shown here.',fontsize=13,va='top')
    fig.suptitle('Soil property distributions — XGBoost',fontsize=19)
    fig.tight_layout();fig.savefig(root/'xgboost_overview.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');args=parser.parse_args()
    publish(project_path(args.directory))


if __name__=='__main__':main()
