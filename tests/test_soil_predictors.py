"""Feature contract checks; not executed during source-only edits."""
import json
import unittest
from module.paths import CONFIG
from module.model_reporting import original_feature


class SoilPredictorTests(unittest.TestCase):
    def test_c_keeps_its_nine_features(self):
        expected={'x','y','elevation','slope','curvature','jshis_avs30','symbol','jshis_jcode','n_value'}
        for target in ('c',):
            config=json.loads((CONFIG/target/'model.jsonc').read_text())
            features=config['numeric_features']+config['categorical_features']
            self.assertEqual(len(features),9)
            self.assertEqual(set(features),expected)
            self.assertNotIn('depth',features)
            self.assertNotIn('surface_z',features)

    def test_sample_elevation_report_label_maps_to_its_own_feature(self):
        self.assertEqual(original_feature('numeric__elevation',['elevation'],[]),'elevation')
