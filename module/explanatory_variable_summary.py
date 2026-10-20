"""Save the c training-data symbol map and counts without retraining."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

from module.paths import project_path
from module.gamma_config import grid_shape
from module.raster import read_binary_mask, boundary_segments


def main():
    source = project_path('data/training/c/model_A/model_dataset.csv')
    output = project_path('data_summary/explanatory_variable/c')
    frame = pd.read_csv(source)
    if frame.empty or not np.isfinite(frame[['x', 'y']].to_numpy()).all():
        raise ValueError('Training data must contain samples with finite coordinates')
    frame['symbol'] = frame.symbol.fillna('UNKNOWN').astype(str)
    sites = frame.groupby(['symbol', 'x', 'y'], dropna=False).agg(
        samples=('target', 'size'), holes=('boring_id', 'nunique')).reset_index()
    if sites.duplicated(['x', 'y']).any():
        raise ValueError('Multiple symbols at one coordinate; resolve before plotting')
    counts = frame.groupby('symbol').agg(
        samples=('target', 'size'), holes=('boring_id', 'nunique'))
    counts['sites'] = sites.groupby('symbol').size()
    counts['percent'] = counts.samples / len(frame) * 100
    legends = []
    for path in sorted(project_path('data/raw/geology/seamlessV2').rglob('legend.tsv')):
        legend = pd.read_csv(path, sep='\t')
        legend['symbol'] = legend.symbol.str.strip()
        legends.append(legend[['symbol', 'formationAge_ja', 'lithology_ja']])
    if legends:
        legend = pd.concat(legends).drop_duplicates('symbol').set_index('symbol')
        counts = counts.join(legend)
    else:
        counts['formationAge_ja'] = ''
        counts['lithology_ja'] = ''
    counts = counts.sort_values('samples', ascending=False)
    colors = {symbol: to_hex(plt.get_cmap('tab10')(i % 10))
              for i, symbol in enumerate(counts.index)}
    counts['color'] = [colors[symbol] for symbol in counts.index]
    preparation_path = source.with_name('preparation_summary.json')
    preparation = json.loads(preparation_path.read_text())
    spatial = preparation['signature']['spatial']
    mask = read_binary_mask(project_path(spatial['property_mask']), grid_shape(spatial))
    segments = boundary_segments(mask, spatial['property_mask_bounds'], spatial['grid_spacing_m'])
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                         'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.add_collection(LineCollection([s / 1000 for s in segments], colors='#939ba3', linewidths=.6))
    for symbol, row in counts.iterrows():
        subset = sites.loc[sites.symbol.eq(symbol)]
        lithology = row.lithology_ja if pd.notna(row.lithology_ja) else '凡例なし'
        ax.scatter(subset.x / 1000, subset.y / 1000, s=42, color=colors[symbol],
                   edgecolors='white', linewidths=.5, zorder=3,
                   label=f'{symbol}：{int(row.samples)}試料 / {int(row.sites)}地点\n{lithology}')
    b = spatial['property_mask_bounds']
    ax.set(xlim=(min(b['xmin'], sites.x.min()) / 1000 - 2,
                 max(b['xmax'], sites.x.max()) / 1000 + 2),
           ylim=(min(b['ymin'], sites.y.min()) / 1000 - 2,
                 max(b['ymax'], sites.y.max()) / 1000 + 2),
           aspect='equal', xlabel='東西方向 [km]', ylabel='南北方向 [km]',
           title=f'c教師データの地質分類（symbol）別分布\n{len(frame)}試料・{len(sites)}地点 ／ {spatial["grid_crs"]}')
    ax.grid(alpha=.18)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, labelspacing=1.2)
    limits = preparation['training_depth_limits']
    fig.text(.05, .025, '前処理後・学習／検証分割前。同一座標の試料は1点に集約。灰色線は解析領域の輪郭。\n'
             f'地質図から付与した地点の分類（試料の土質分類ではありません）。'
             f'学習深度条件：{limits["training_min_depth_m"]}〜{limits["training_max_depth_m"]} m。', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'c_symbol_map.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    counts.to_csv(output / 'c_symbol_counts.csv', encoding='utf-8-sig')
    sites.to_csv(output / 'c_symbol_sites.csv', index=False, encoding='utf-8-sig')
    summary = dict(source=str(source.relative_to(project_path('.'))),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   scope='prepared c data before train/validation/test splitting',
                   samples=len(frame), holes=int(frame.boring_id.nunique()), sites=len(sites),
                   symbols=len(counts), missing_symbol_samples=int(frame.symbol.eq('UNKNOWN').sum()),
                   training_depth_limits=limits, crs=spatial['grid_crs'], colors=colors)
    (output / 'c_symbol_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
