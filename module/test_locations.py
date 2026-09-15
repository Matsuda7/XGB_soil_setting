"""Plot measured soil-test sites against the full-resolution prop zero boundary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import contourpy
import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from module.paths import CONFIG, project_path

DATUM_CRS = {'00': 'EPSG:4301', '01': 'EPSG:4612', '02': 'EPSG:6668'}
C_FIELDS = ['c_total', 'c_effective']
STRENGTH_FIELDS = C_FIELDS + ['shear_strength_total', 'shear_strength_effective', 'qu']


def read_binary_mask(path, shape):
    """Stream rows to a one-byte mask; retain every zero/nonzero transition."""
    mask = np.empty(shape, dtype=np.uint8)
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            if count >= shape[0]:
                raise ValueError('prop has more rows than configured')
            values = np.fromstring(line, sep=' ')
            if values.size != shape[1] or not np.isfinite(values).all():
                raise ValueError(f'Invalid prop row {count + 1}: expected {shape[1]} finite values')
            mask[count] = values != 0
            count += 1
    if count != shape[0]:
        raise ValueError(f'prop has {count} rows, expected {shape[0]}')
    return mask


def boundary_segments(mask, bounds, spacing, block_rows=256):
    """March at level 0.5 in overlapping strips without downsampling the grid."""
    segments = []
    x = float(bounds['xmin']) + np.arange(mask.shape[1]) * spacing
    for start in range(0, mask.shape[0] - 1, block_rows):
        stop = min(start + block_rows + 1, mask.shape[0])
        z = mask[start:stop]
        if z.min() == z.max():
            continue
        y = float(bounds['ymin']) + np.arange(start, stop) * spacing
        generator = contourpy.contour_generator(x=x, y=y, z=z, name='serial')
        segments.extend(generator.lines(0.5))
    return segments


def load_test_records(xml_dir):
    """Include density-only tests omitted by the strength-oriented CSV extractor."""
    from module.strength import context_values, make_row, candidate_records
    rows = []
    for path in sorted(Path(xml_dir).rglob('*.xml')):
        content = path.read_bytes()
        try:
            text = content.decode('shift_jis')
        except UnicodeDecodeError:
            text = content.decode('utf-8-sig')
        root = ET.fromstring(text)
        context = context_values(root)
        records = [e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == '試験情報']
        for record in records or candidate_records(root):
            rows.append(make_row(context, record, path.name))
    if not rows:
        raise ValueError('No soil-test records found in source XML')
    return pd.DataFrame(rows)


def has_values(frame, columns):
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors='coerce')
    return pd.Series(np.isfinite(values.to_numpy()).any(axis=1), index=frame.index)


def build_sites(frame, xml_dir, target_crs):
    """Respect XML datum codes, aggregate repeated samples per borehole location."""
    frame = frame.copy()
    frame['has_c'] = has_values(frame, C_FIELDS)
    frame['c_related'] = has_values(frame, STRENGTH_FIELDS)
    frame['wet'] = has_values(frame, ['wet_density'])
    frame['dry'] = has_values(frame, ['dry_density'])
    frame = frame.loc[frame[['c_related', 'wet', 'dry']].any(axis=1)].copy()
    codes = {}
    for filename in frame.xml_file.unique():
        matches = list(Path(xml_dir).rglob(str(filename)))
        if len(matches) != 1:
            raise ValueError(f'Expected one source XML for {filename}, found {len(matches)}')
        content = matches[0].read_bytes()
        try:
            text = content.decode('shift_jis')
        except UnicodeDecodeError:
            text = content.decode('utf-8-sig')
        root = ET.fromstring(text)
        datum = {str(e.text).strip().zfill(2) for e in root.iter() if e.tag.rsplit('}', 1)[-1] == '測地系'}
        if len(datum) != 1 or next(iter(datum)) not in DATUM_CRS:
            raise ValueError(f'Unknown or ambiguous datum for {filename}: {datum}')
        codes[filename] = next(iter(datum))
    frame['datum_code'] = frame.xml_file.map(codes)
    lon = pd.to_numeric(frame.longitude, errors='coerce')
    lat = pd.to_numeric(frame.latitude, errors='coerce')
    valid = lon.between(-180, 180) & lat.between(-90, 90)
    excluded = frame.loc[~valid].copy()
    frame = frame.loc[valid].copy()
    frame['x'] = np.nan
    frame['y'] = np.nan
    for code, group in frame.groupby('datum_code'):
        transformer = Transformer.from_crs(DATUM_CRS[code], target_crs, always_xy=True)
        x, y = transformer.transform(lon.loc[group.index].to_numpy(dtype=float), lat.loc[group.index].to_numpy(dtype=float))
        frame.loc[group.index, 'x'] = x
        frame.loc[group.index, 'y'] = y
    if not np.isfinite(frame[['x', 'y']].to_numpy()).all():
        raise ValueError('Non-finite projected coordinates')
    sites = frame.groupby(['boring_id', 'x', 'y'], dropna=False).agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first'),
        datum_code=('datum_code', 'first'), records=('xml_file', 'size'),
        has_c=('has_c', 'any'), c_related=('c_related', 'any'), wet=('wet', 'any'), dry=('dry', 'any'),
    ).reset_index()
    return sites, excluded


def create_map(config_path, output_dir=None):
    options = json.loads(Path(config_path).read_text(encoding='utf-8'))
    config = json.loads(project_path(options['prediction_config']).read_text(encoding='utf-8'))
    output = Path(output_dir) if output_dir is not None else project_path(options['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    bounds = config['property_mask_bounds']
    spacing = float(config['property_mask_spacing_m'])
    if spacing <= 0:
        raise ValueError('Mask spacing must be positive')
    shape = (int(round((bounds['ymax']-bounds['ymin'])/spacing))+1,
             int(round((bounds['xmax']-bounds['xmin'])/spacing))+1)
    frame = load_test_records(project_path(options['xml_dir']))
    sites, excluded = build_sites(frame, project_path(options['xml_dir']), config['grid_crs'])
    mask = read_binary_mask(project_path(config['property_mask']), shape)
    segments = boundary_segments(mask, bounds, spacing)
    sites['inside_prop_extent'] = sites.x.between(bounds['xmin'], bounds['xmax']) & sites.y.between(bounds['ymin'], bounds['ymax'])
    sites.to_csv(output/'test_locations.csv', index=False, encoding='utf-8-sig')
    excluded.to_csv(output/'excluded_records.csv', index=False, encoding='utf-8-sig')
    counts = {key:int(sites[key].sum()) for key in ('c_related','has_c','wet','dry')}
    gamma = sites.wet | sites.dry
    counts['gamma'] = int(gamma.sum())
    counts['outside_prop_extent'] = int((~sites.inside_prop_extent).sum())
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'axes.unicode_minus':False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    selections = [sites.c_related, gamma]
    colors = ['#cd682b', '#1769aa']
    titles = [f'c関連の強度試験：{counts["c_related"]}地点', f'γの密度試験：{counts["gamma"]}地点']
    xmin = min(bounds['xmin'], sites.x.min()) - 2000
    xmax = max(bounds['xmax'], sites.x.max()) + 2000
    ymin = min(bounds['ymin'], sites.y.min()) - 2000
    ymax = max(bounds['ymax'], sites.y.max()) + 2000
    for ax, selected, color, title in zip(axes, selections, colors, titles):
        ax.set_facecolor('#f7f9fb')
        ax.add_collection(LineCollection([segment / 1000 for segment in segments], colors='#343f48', linewidths=0.65))
        points = sites.loc[selected]
        inside = points.inside_prop_extent
        ax.scatter(points.loc[inside, 'x']/1000, points.loc[inside, 'y']/1000, s=25, c=color, edgecolors='white', linewidths=0.4, zorder=3, label='試験地点')
        if (~inside).any():
            ax.scatter(points.loc[~inside, 'x']/1000, points.loc[~inside, 'y']/1000, s=28, facecolors='none', edgecolors=color, linewidths=1, zorder=3, label='propの範囲外の試験地点')
        ax.plot([], [], color='#343f48', lw=1, label='prop=0 / prop≠0 の境界')
        ax.set(xlim=(xmin/1000,xmax/1000), ylim=(ymin/1000,ymax/1000), aspect='equal', title=title,
               xlabel='東西方向 [km]', ylabel='南北方向 [km]')
        ax.grid(alpha=0.18)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
    axes[0].set_title(titles[0] + f'\n粘着力cそのものの値がある地点：{counts["has_c"]}', fontsize=11)
    axes[1].set_title(titles[1] + f'\n湿潤密度：{counts["wet"]}地点 ／ 乾燥密度：{counts["dry"]}地点', fontsize=11)
    fig.supxlabel('同じ孔の複数試料は1地点として表示。c関連はせん断強さ・一軸圧縮強さ・粘着力の記録を対象。', fontsize=9)
    fig.suptitle('能登半島周辺の土質試験地点\n' + config['grid_crs'] + ' ／ 輪郭はpropの0・非0境界（予測分布ではありません）', fontsize=15)
    fig.savefig(output/'c_gamma_test_locations.png', dpi=200, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(output/'c_gamma_test_locations.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    summary = {'input_source':'raw XML including density-only tests', 'sites':counts,'excluded_records':len(excluded), 'mask_shape':list(mask.shape),
               'boundary':'full-resolution zero/nonzero contour at binary level 0.5',
               'crs':config['grid_crs'], 'datum_codes':DATUM_CRS,
               'datum_reference':'https://www.maff.go.jp/j/nousin/seko/nouhin_youryou/attach/pdf/doboku-40.pdf'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG/'test_locations.json')
    args = parser.parse_args(argv)
    create_map(project_path(args.config))
    return 0
