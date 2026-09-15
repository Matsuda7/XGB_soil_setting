"""One predictor-building path for gamma observations and unseen grid points."""
import numpy as np
import pandas as pd
from pyproj import Transformer
from module.paths import project_path
from module.gamma_config import grid_shape
from module.terrain import load_or_create_dem_cache, sample_terrain_features
from module.spatial_features import load_geology, load_regional_jshis, attach_spatial_attributes
from module.xgb_common import prepare_feature_frame


def load_dem(config):
    spatial = config['spatial']
    return load_or_create_dem_cache(project_path(spatial['dem_input']), project_path(spatial['dem_cache']), grid_shape(spatial))


def terrain_at(points, dem, config):
    spatial = config['spatial']; bounds = spatial['property_mask_bounds']
    return sample_terrain_features(points, dem, xmin=bounds['xmin'], ymin=bounds['ymin'], spacing=spatial['grid_spacing_m'])


def geographic_context(config, extra_points=None):
    bounds = config['spatial']['property_mask_bounds']
    x,y = np.meshgrid(np.linspace(bounds['xmin'],bounds['xmax'],5), np.linspace(bounds['ymin'],bounds['ymax'],5))
    x,y = x.ravel(),y.ravel()
    if extra_points is not None:
        x=np.concatenate([x,extra_points.x.to_numpy()]); y=np.concatenate([y,extra_points.y.to_numpy()])
    transformer = Transformer.from_crs(config['spatial']['grid_crs'],'EPSG:4326',always_xy=True)
    lon,lat = transformer.transform(x,y)
    prefixes = {f'{a:02d}{b:02d}' for a in range(int(np.floor(min(lat)*1.5)),int(np.floor(max(lat)*1.5))+1)
                for b in range(int(np.floor(min(lon)-100)),int(np.floor(max(lon)-100))+1)}
    return load_geology(), load_regional_jshis(prefixes)


def enrich(points, config, dem, geology, jshis):
    frame = points.reset_index(drop=True).copy()
    terrain = terrain_at(frame, dem, config)
    frame[terrain.columns] = terrain
    frame['sample_z'] = frame.surface_z-frame.depth
    return attach_spatial_attributes(frame, geology, jshis, config['spatial']['grid_crs'])


def model_frame(frame, numeric, categorical):
    frame = frame.copy()
    # Integer codes must have identical strings whether a chunk contains missing values or not.
    for column in set(categorical) & {'jshis_jcode','ser'}:
        frame[column] = pd.to_numeric(frame[column],errors='coerce').astype('Int64').astype('string')
    return prepare_feature_frame(frame,numeric,categorical)
