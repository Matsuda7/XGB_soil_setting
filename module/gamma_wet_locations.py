"""Map measured wet density and records with water content plus full grain fractions."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from module.paths import project_path
from module.soil_tests import load_test_records, project_records
from module.gamma_config import grid_shape
from module.raster import read_binary_mask, boundary_segments

GRAINS = ['gravel_fraction', 'sand_fraction', 'silt_fraction', 'clay_fraction']


def classify_records(frame):
    result = frame.copy()
    density = pd.to_numeric(result.wet_density, errors='coerce')
    water = pd.to_numeric(result.water_content, errors='coerce')
    grains = result[GRAINS].apply(pd.to_numeric, errors='coerce')
    result['measured_wet'] = np.isfinite(density) & density.gt(0)
    complete_grains = np.isfinite(grains).all(axis=1) & grains.ge(0).all(axis=1) & grains.le(100).all(axis=1)
    result['candidate'] = density.isna() & np.isfinite(water) & water.ge(0) & complete_grains
    return result


def aggregate_sites(frame):
    selected = frame.loc[frame.measured_wet | frame.candidate]
    return selected.groupby(['x', 'y'], sort=True).agg(
        measured_records=('measured_wet', 'sum'), candidate_records=('candidate', 'sum'),
        boreholes=('boring_id', 'nunique')).reset_index()


def create_map(output='results/test_locations/gamma_wet_availability'):
    output = project_path(output); output.mkdir(parents=True, exist_ok=True)
    spatial = json.loads(project_path('config/spatial.json').read_text())
    raw = classify_records(load_test_records(project_path('data/raw/soiltest/strength')))
    # Use original measurements, including records outside the model's DEM/depth filters.
    selected = raw.loc[raw.measured_wet | raw.candidate].copy()
    frame, excluded = project_records(selected, project_path('data/raw/soiltest/strength'), spatial['grid_crs'])
    sites = aggregate_sites(frame)
    wet = sites.measured_records.gt(0); candidate = sites.candidate_records.gt(0)
    summary = dict(source='raw soil-test XML; not restricted to gamma training population',
        crs=spatial['grid_crs'], raw_records=len(raw), plotted_records=len(frame),
        measured_records=int(frame.measured_wet.sum()), candidate_records=int(frame.candidate.sum()),
        measured_sites=int(wet.sum()), candidate_sites=int(candidate.sum()),
        overlapping_sites=int((wet & candidate).sum()), unique_sites=len(sites),
        excluded_coordinate_records=len(excluded),
        candidate_definition='wet_density missing AND water_content finite >=0 AND gravel/sand/silt/clay all finite within [0,100]',
        overlap_display='cyan outer circle with orange center; different samples at same XY',
        colors={'measured':'#ef8a23','candidate':'#67c9ec'})
    frame.to_csv(output/'selected_records.csv',index=False,encoding='utf-8-sig')
    sites.to_csv(output/'sites.csv',index=False,encoding='utf-8-sig')
    excluded.to_csv(output/'excluded_records.csv',index=False,encoding='utf-8-sig')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    mask = read_binary_mask(project_path(spatial['property_mask']), grid_shape(spatial))
    segments = boundary_segments(mask, spatial['property_mask_bounds'], spatial['grid_spacing_m'])
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(10,11),constrained_layout=True)
    ax.set_facecolor('white')
    ax.add_collection(LineCollection([s/1000 for s in segments],colors='#57616b',linewidths=.7))
    ax.scatter(sites.loc[candidate,'x']/1000,sites.loc[candidate,'y']/1000,s=74,c='#67c9ec',
        edgecolors='#348da9',linewidths=.5,zorder=3,label=f'湿潤密度なし・含水比＋粒度あり：{int(candidate.sum())}地点')
    ax.scatter(sites.loc[wet,'x']/1000,sites.loc[wet,'y']/1000,s=30,c='#ef8a23',
        edgecolors='white',linewidths=.5,zorder=4,label=f'湿潤密度の実測あり：{int(wet.sum())}地点')
    b=spatial['property_mask_bounds']
    ax.set(xlim=(min(b['xmin'],sites.x.min())/1000-2,max(b['xmax'],sites.x.max())/1000+2),
        ylim=(min(b['ymin'],sites.y.min())/1000-2,max(b['ymax'],sites.y.max())/1000+2),
        aspect='equal',xlabel='東西方向 [km]',ylabel='南北方向 [km]',
        title='湿潤密度の実測地点と追加候補地点\n'+spatial['grid_crs'])
    ax.grid(alpha=.17);ax.legend(loc='upper left',fontsize=9,framealpha=.97)
    fig.supxlabel('粒度あり＝礫・砂・シルト・粘土の4成分が同じ試料に存在。\n'
        f'同一座標の複数試料は集約。両条件がある{summary["overlapping_sites"]}地点は水色の外周＋オレンジの中心で表示。\n'
        '輪郭は解析対象範囲。水色は追加候補であり、今回のγ学習には使用しません。',fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(output/f'gamma_wet_locations.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/test_locations/gamma_wet_availability')
    args=parser.parse_args();create_map(args.output)


if __name__=='__main__':
    main()
