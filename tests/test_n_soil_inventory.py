import unittest
from unittest.mock import patch
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from module.solphyj_features import attach_soil_inventory, attach_solphyj, build_n_preprocessor, SOIL_CATEGORY, COLUMNS
from module.xgb_common import load_model_config
from module.paths import CONFIG


class SoilInventoryTests(unittest.TestCase):
    def setUp(self):
        self.soil = gpd.GeoDataFrame({'SSerGrCD': ['A', 'B', 'A']},
            geometry=[box(0,0,2,2), box(1,0,3,2), box(0,0,1,2)], crs='EPSG:6675')
        self.frame = pd.DataFrame({'x': [.5,1.5,4.,0.], 'y': [1.,1.,1.,1.]}, index=[9,2,7,3])

    def test_codes_preserve_rows_and_reject_conflicting_polygons(self):
        with patch('module.solphyj_features.load_soil_lookup', return_value=self.soil):
            result = attach_soil_inventory(self.frame, 'EPSG:6675')
        self.assertEqual(result.index.tolist(), self.frame.index.tolist())
        self.assertEqual(result[SOIL_CATEGORY].tolist(), ['A','UNKNOWN','UNKNOWN','UNKNOWN'])
        self.assertFalse(set(COLUMNS) & set(result.columns))

    def test_unseen_category_does_not_expand_fitted_encoding(self):
        pre = build_n_preprocessor(['depth'], [SOIL_CATEGORY])
        pre.fit(pd.DataFrame({'depth':[1.,2.], SOIL_CATEGORY:['A','B']}))
        transformed = pre.transform(pd.DataFrame({'depth':[3.], SOIL_CATEGORY:['C']}))
        np.testing.assert_allclose(transformed, [[3.,0.,0.]])

    def test_legacy_texture_lookup_still_works(self):
        props = pd.DataFrame({c:[20.,30.] for c in COLUMNS}, index=['A','B'])
        with patch('module.solphyj_features.load_soil_lookup', return_value=self.soil), patch('module.solphyj_features.load_lookup', return_value=(self.soil,props)):
            result = attach_solphyj(self.frame, 'EPSG:6675')
        self.assertEqual(result[COLUMNS[0]].iloc[0], 20.)
        self.assertTrue(result[COLUMNS].iloc[1:].isna().all(axis=None))

    def test_active_config_uses_inventory_instead_of_texture(self):
        numeric, categorical = load_model_config(CONFIG/'phi/model.jsonc')
        self.assertIn(SOIL_CATEGORY, categorical)
        self.assertFalse(set(COLUMNS) & set(numeric))
