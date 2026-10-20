"""Map all geological symbols in the configured analysis bounding box."""
import json

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import pandas as pd
from shapely.geometry import box

from module.paths import project_path
from module.spatial_features import load_geology, GEOLOGY_DIR


def main():
    spatial = json.loads(project_path('config/spatial.json').read_text())
    b = spatial['property_mask_bounds']
    bounds = box(b['xmin'], b['ymin'], b['xmax'], b['ymax'])
    geology = load_geology().to_crs(spatial['grid_crs'])
    geology = gpd.clip(geology, bounds, keep_geom_type=True)
    geology = geology.loc[~geology.geometry.is_empty].copy()
    geology['symbol'] = geology.symbol.str.strip()
    if geology.empty:
        raise ValueError('No geology polygons within analysis bounds')
    legends = []
    for path in sorted(GEOLOGY_DIR.rglob('legend.tsv')):
        frame = pd.read_csv(path, sep='\t')
        frame['symbol'] = frame.symbol.str.strip()
        legends.append(frame)
    legend = pd.concat(legends).drop_duplicates('symbol').set_index('symbol')
    symbols = sorted(geology.symbol.unique())
    legend = legend.reindex(symbols)
    colors = {}
    for symbol, row in legend.iterrows():
        colors[symbol] = (to_hex(tuple(row[['r', 'g', 'b']].astype(float) / 255))
                          if row[['r', 'g', 'b']].notna().all() else '#bdbdbd')
    legend['color'] = [colors[s] for s in symbols]
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9,
                         'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(14, 11))
    # The repository uses easting as x and northing as y (always_xy).
    display = geology.copy()
    display.geometry = display.geometry.scale(xfact=.001, yfact=.001, origin=(0, 0))
    display.plot(ax=ax, color=display.symbol.map(colors), edgecolor='none', rasterized=True)
    ax.set(xlim=(b['xmin']/1000, b['xmax']/1000),
           ylim=(b['ymin']/1000, b['ymax']/1000), aspect='equal',
           xlabel='東西方向 [km]', ylabel='南北方向 [km]',
           title=f'地質分類 symbol の全体分布（{len(symbols)}分類）\n{spatial["grid_crs"]}')
    ax.grid(alpha=.12)
    handles = [Patch(facecolor=colors[s], edgecolor='#888888', linewidth=.3,
                     label=f'{s}  {legend.loc[s, "lithology_ja"]}') for s in symbols]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=7, frameon=False)
    fig.text(.04, .02, '出典：産総研 地質調査総合センター 20万分の1日本シームレス地質図V2（ローカル取得データ）。\n'
             'config/spatial.jsonの矩形範囲で切り出し。教師データの有無で限定せず全地質分類を表示。白地はポリゴンなし。', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    output = project_path('data_summary/explanatory_variable/symbol')
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output/f'symbol_distribution.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    legend[['formationAge_ja', 'lithology_ja', 'color']].to_csv(output/'symbol_legend.csv', encoding='utf-8-sig')
    summary = dict(source='data/raw/geology/seamlessV2/**/*_poly.shp',
                   scope='all geology polygons clipped to configured rectangular bounds; not prop-mask clipped',
                   bounds=b, crs=spatial['grid_crs'], symbol_count=len(symbols),
                   symbols=symbols, polygon_count=len(geology),
                   colors='GSJ legend.tsv RGB; gray if unavailable')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
