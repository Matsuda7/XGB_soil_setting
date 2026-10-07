"""Shared SSG texture lookup for N-value training and grid prediction."""
from functools import lru_cache
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from module.paths import project_path
from module.xgb_common import build_preprocessor

COLUMNS = ['representative_total_sand', 'representative_silt', 'representative_clay']
SOIL_CATEGORY = 'soil_inventory_ssg'


@lru_cache(maxsize=4)
def load_soil_lookup(crs):
    settings = json.loads(project_path('config/phi/solphyj.json').read_text())
    soil = gpd.read_file('zip://' + str(project_path(settings['soil_archive'])), encoding='cp932').to_crs(crs)
    soil['SSerGrCD'] = soil.SSerGrCD.astype('string')
    return soil[['SSerGrCD', 'geometry']]


@lru_cache(maxsize=4)
def load_lookup(crs):
    settings = json.loads(project_path('config/phi/solphyj.json').read_text())
    soil = load_soil_lookup(crs)
    props = pd.read_csv(project_path(settings['representatives']), dtype={'SSG code': str})
    if props['SSG code'].duplicated().any():
        raise ValueError('Duplicate SolphyJ SSG codes')
    return soil[['SSerGrCD', 'geometry']], props.set_index('SSG code')[COLUMNS]


def soil_codes(frame, crs):
    """Return map codes in input row order; conflicting polygons stay missing."""
    soil = load_soil_lookup(crs)
    points = gpd.GeoDataFrame({'_soil_row': np.arange(len(frame))},
        geometry=gpd.points_from_xy(frame.x, frame.y), crs=crs)
    joined = gpd.sjoin(points, soil, how='left', predicate='within')
    counts = joined.groupby('_soil_row').SSerGrCD.nunique()
    ambiguous = counts[counts > 1].index
    joined = joined.sort_values(['_soil_row', 'index_right']).drop_duplicates('_soil_row')
    joined.loc[joined._soil_row.isin(ambiguous), 'SSerGrCD'] = pd.NA
    codes = joined.set_index('_soil_row').SSerGrCD.reindex(range(len(frame)))
    return codes


def attach_soil_inventory(frame, crs):
    result = frame.copy()
    result[SOIL_CATEGORY] = soil_codes(frame, crs).fillna('UNKNOWN').to_numpy()
    return result


def attach_solphyj(frame, crs):
    """Preserve historical numeric features for previously saved models."""
    _, props = load_lookup(crs)
    result = frame.copy()
    codes = soil_codes(frame, crs)
    for column in COLUMNS:
        result[column] = codes.map(props[column]).to_numpy(dtype=float, na_value=np.nan)
    return result


def build_n_preprocessor(numeric, categorical):
    """Retain existing preprocessing; let XGBoost handle SolphyJ missing values."""
    extra = [c for c in numeric if c in COLUMNS]
    pre = build_preprocessor([c for c in numeric if c not in COLUMNS], categorical)
    if extra:
        pre.transformers.append(('solphyj', 'passthrough', extra))
    return pre
