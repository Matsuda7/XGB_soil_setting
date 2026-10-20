import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.gamma_config import load_config
from module.gamma_features import enrich
from module.gamma_wet_locations import classify_records, aggregate_sites
from module.paths import CONFIG
from module.xgb_common import load_model_config


class GammaMeasuredPhiTests(unittest.TestCase):
    def test_measured_config_matches_phi_without_n(self):
        config=load_config()
        numeric,categorical=load_model_config(CONFIG/'phi/model.jsonc')
        self.assertEqual(config['numeric_features'],numeric)
        self.assertEqual(config['categorical_features'],categorical)
        self.assertEqual(config['gamma_kind'],'wet')
        self.assertEqual(config['distribution_model'],'no_n')
        self.assertFalse(config['n_comparison']['enabled'])
        self.assertNotIn('n_value',numeric)

    def test_enrich_uses_sample_elevation_and_inventory(self):
        terrain=pd.DataFrame({'surface_z':[10.], 'slope':[0.], 'curvature':[0.]})
        config={'categorical_features':['soil_inventory_ssg'],'spatial':{'grid_crs':'EPSG:6675'}}
        with patch('module.gamma_features.terrain_at',return_value=terrain), \
             patch('module.gamma_features.attach_spatial_attributes',side_effect=lambda frame,*args:frame), \
             patch('module.solphyj_features.attach_soil_inventory',side_effect=lambda frame,*args:frame.assign(soil_inventory_ssg='A')):
            result=enrich(pd.DataFrame({'x':[1.],'y':[2.],'depth':[3.]}),config,None,None,None)
        self.assertEqual(result.sample_elevation.iloc[0],7.)
        self.assertEqual(result.soil_inventory_ssg.iloc[0],'A')

    def test_candidates_require_water_and_all_grains_in_same_record(self):
        f=pd.DataFrame({'wet_density':[1.8,np.nan,np.nan,np.nan,0],
            'water_content':[np.nan,20,np.nan,20,20],
            'gravel_fraction':[0]*5,'sand_fraction':[30]*5,'silt_fraction':[30]*5,
            'clay_fraction':[40,40,40,np.nan,40], 'x':[1]*5,'y':[2]*5,'boring_id':['A']*5})
        marked=classify_records(f)
        self.assertEqual(marked.measured_wet.tolist(),[True,False,False,False,False])
        self.assertEqual(marked.candidate.tolist(),[False,True,False,False,False])
        sites=aggregate_sites(marked)
        self.assertEqual(len(sites),1)
        self.assertEqual(sites.measured_records.iloc[0],1)
        self.assertEqual(sites.candidate_records.iloc[0],1)
