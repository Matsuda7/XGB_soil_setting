"""Measured wet gamma and water-content/grain-size sites within the training DEM."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from module.paths import project_path
from module.soil_tests import load_test_records, project_records
from module.gamma_config import load_config, grid_shape
from module.gamma_features import load_dem, terrain_at
from module.gamma_data import density_targets
from module.raster import read_binary_mask, boundary_segments


def main():
    config = load_config()
    spatial = config['spatial']
    output = project_path('results/test_locations/gamma_water_grains_training_area')
    output.mkdir(parents=True, exist_ok=True)
    source = project_path('data/raw/soiltest/strength')
    raw = density_targets(load_test_records(source), config['gravity_m_s2'])
    raw['depth'] = pd.to_numeric(raw.depth_mid, errors='coerce')
    raw['wet'] = raw.gamma_wet.notna()
    water = pd.to_numeric(raw.water_content, errors='coerce')
    raw['water'] = np.isfinite(water) & water.ge(0)
    grains = raw[['gravel_fraction','sand_fraction','silt_fraction','clay_fraction']].apply(pd.to_numeric, errors='coerce')
    raw['grains'] = (np.isfinite(grains) & grains.ge(0) & grains.le(100)).any(axis=1)
    raw['other'] = raw.water | raw.grains
    selected = raw.loc[(raw.wet | raw.other) & np.isfinite(raw.depth) & raw.depth.ge(0)].copy()
    selected = selected.drop_duplicates(['xml_file','sample_id','sample_no','depth','wet_density','dry_density'])
    frame, bad_coordinates = project_records(selected, source, spatial['grid_crs'])
    terrain = terrain_at(frame, load_dem(config), config)
    inside = terrain.surface_z.notna().to_numpy()
    excluded = frame.loc[~inside].copy()
    frame = frame.loc[inside].copy()
    sites = frame.groupby(['x','y'], sort=True).agg(wet=('wet','any'), other=('other','any'),
        water=('water','any'), grains=('grains','any'), boreholes=('boring_id','nunique')).reset_index()
    summary = dict(scope='valid-depth records inside training DEM; all depths',
        other_definition='water content OR at least one valid gravel/sand/silt/clay percentage',
        wet_sites=int(sites.wet.sum()), wet_records=int(frame.wet.sum()),
        other_sites=int(sites.other.sum()), other_records=int(frame.other.sum()),
        water_sites=int(sites.water.sum()), grain_sites=int(sites.grains.sum()),
        overlapping_sites=int((sites.wet & sites.other).sum()), total_sites=len(sites),
        outside_dem_records=len(excluded), invalid_coordinate_records=len(bad_coordinates))
    frame.to_csv(output/'records.csv', index=False, encoding='utf-8-sig')
    sites.to_csv(output/'sites.csv', index=False, encoding='utf-8-sig')
    excluded.to_csv(output/'outside_dem.csv', index=False, encoding='utf-8-sig')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    mask = read_binary_mask(project_path(spatial['property_mask']), grid_shape(spatial))
    bounds = spatial['property_mask_bounds']; spacing = spatial['grid_spacing_m']
    segments = boundary_segments(mask,bounds,spacing)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False})
    fig,ax = plt.subplots(figsize=(9,10))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    extent=[(bounds['xmin']-spacing/2)/1000,(bounds['xmax']+spacing/2)/1000,
            (bounds['ymin']-spacing/2)/1000,(bounds['ymax']+spacing/2)/1000]
    ax.imshow(mask,origin='lower',extent=extent,cmap=ListedColormap(['white','#d4d4d4']),
              vmin=0,vmax=1,interpolation='nearest',zorder=0)
    ax.add_collection(LineCollection([segment/1000 for segment in segments],colors='#242424',linewidths=.8,zorder=2))
    other=sites.loc[sites.other]; wet=sites.loc[sites.wet]
    ax.scatter(other.x/1000,other.y/1000,s=43,c='#67c9ec',edgecolors='#347d97',linewidths=.5,zorder=4,
               label=f'含水比・粒度分布の試験：{len(other)}地点')
    ax.scatter(wet.x/1000,wet.y/1000,s=20,c='#ef8a23',edgecolors='#865114',linewidths=.4,zorder=5,
               label=f'湿潤単位体積重量の実測：{len(wet)}地点')
    ax.set(xlim=(bounds['xmin']/1000-1,bounds['xmax']/1000+1),
           ylim=(bounds['ymin']/1000-1,bounds['ymax']/1000+1),aspect='equal',
           xlabel='東西方向 [km]',ylabel='南北方向 [km]',title='土質試験の実施地点（モデル学習対象範囲内）')
    ax.grid(color='#ababab',linestyle=':',alpha=.45,zorder=1)
    ax.legend(loc='upper left',fontsize=10,framealpha=.95)
    fig.text(.5,.025,'含水比または粒度分布の記録がある地点を水色で表示。同一座標の試料は集約。\n'
        f'両方に該当する{summary["overlapping_sites"]}地点は水色の外周＋オレンジの中心。灰色は解析領域。',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,1))
    for ext in ['png','pdf']:
        fig.savefig(output/f'soil_test_locations.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
