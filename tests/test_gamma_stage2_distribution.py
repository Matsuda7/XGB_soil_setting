import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from module.gamma_stage2_distribution import FeatureBuilder, attach_soil, load_config
from module.gamma_stage2_integrated import SOLPHYJ_COLUMNS


class DepthDistributionTests(unittest.TestCase):
    def test_f3_configuration_has_depth_without_geology_or_jshis(self):
        config = load_config('config/gamma/stage2_f3_distribution.json')
        self.assertEqual(config['distribution_model'], 'F3_depth')
        self.assertEqual(config['numeric_features'], ['x', 'y', 'elevation', 'depth', *SOLPHYJ_COLUMNS])
        self.assertEqual(config['categorical_features'], [])

    def test_depth_validation(self):
        from module.configuration import _read_json
        for depth in (-1, float('nan'), True, '1', 2.5):
            def read(path):
                if path == 'config/model.json':
                    return {'prediction_depth_m': depth, 'observed_depth_tolerance_m': 1.0}
                return _read_json(path)
            with patch('module.configuration._read_json', side_effect=read):
                if depth == 2.5:
                    self.assertEqual(load_config('config/gamma/stage2_distribution.json')['prediction_depth_m'], 2.5)
                else:
                    with self.assertRaises(ValueError):
                        load_config('config/gamma/stage2_distribution.json')

    def test_soil_overlap_and_outside_remain_missing(self):
        soil = gpd.GeoDataFrame({'SSerGrCD': ['A', 'B']}, geometry=[box(0,0,2,2),box(1,0,3,2)], crs='EPSG:6675')
        reps = pd.DataFrame({'SSG code': ['A', 'B'], **{col: [10.,20.] for col in SOLPHYJ_COLUMNS}})
        points = pd.DataFrame({'x': [.5,1.5,4], 'y': [1.,1.,1.]})
        result = attach_soil(points, soil, reps, 'EPSG:6675')
        self.assertEqual(result.representative_clay.iloc[0], 10.)
        self.assertTrue(result.representative_clay.iloc[1:].isna().all())
        self.assertEqual(len(result), len(points))

    def test_depth_changes_elevation_with_same_surface_attributes(self):
        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.config = {}; builder.crs = 'EPSG:6675'; builder.dem = None
        builder.geology = gpd.GeoDataFrame({'symbol': ['G']}, geometry=[box(0,0,3,3)], crs=builder.crs)
        builder.soil = gpd.GeoDataFrame({'SSerGrCD': ['A']}, geometry=[box(0,0,3,3)], crs=builder.crs)
        builder.representative = pd.DataFrame({'SSG code': ['A'], **{col:[30.] for col in SOLPHYJ_COLUMNS}})
        builder.jshis = pd.DataFrame({'jshis_meshcode':['mesh'], 'jshis_avs30':[200.], 'jshis_jcode':[1.]})
        class Transform:
            def transform(self, x, y):
                return x, y
        builder.transform = Transform()
        terrain = pd.DataFrame({'surface_z':[10.,10.], 'slope':[1.,1.], 'curvature':[0.,0.]})
        with patch('module.gamma_stage2_distribution.terrain_at', return_value=terrain), \
             patch('module.gamma_stage2_distribution.meshcodes_250m', return_value=['mesh','mesh']):
            result = builder(pd.DataFrame({'x':[1.,1.], 'y':[1.,1.], 'depth':[1.,4.]}))
        np.testing.assert_allclose(result.elevation, [9.,6.])
        np.testing.assert_allclose(result.depth, [1.,4.])
        self.assertEqual(result.geology.tolist(), ['G','G'])
        np.testing.assert_allclose(result.representative_clay, [30.,30.])


if __name__ == '__main__':
    unittest.main()
