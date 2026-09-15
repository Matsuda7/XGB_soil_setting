"""Shared complete soil-test extraction and datum-aware projection."""
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from pyproj import Transformer
DATUM_CRS = {'00': 'EPSG:4301', '01': 'EPSG:4612', '02': 'EPSG:6668'}

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


def project_records(frame, xml_dir, target_crs):
    frame = frame.copy()
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
    return frame, excluded
