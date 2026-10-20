import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from module.observed_overlay import select_observations, add_overlay
from module.configuration import load_prediction_settings


class ObservedOverlayTests(unittest.TestCase):
    def test_inclusive_window_nearest_ties_and_invalid_values(self):
        f = pd.DataFrame({'x':[0,0,0,1,2,3,4,5], 'y':[0]*8,
            'depth':[4,6,4.5,4,6,6.01,5,5], 'value':[10,20,30,40,50,60,np.nan,-1]})
        p=select_observations(f,5,1,[-1,6,-1,1])
        self.assertEqual(p.x.tolist(),[0,1,2])
        self.assertEqual(p.value.tolist(),[30,40,50])
        ties=select_observations(f.iloc[:2],5,1,[-1,6,-1,1])
        self.assertEqual(ties.value.tolist(),[15])
        self.assertEqual(ties.samples.tolist(),[2])
        self.assertTrue(select_observations(f,10,1,[-1,6,-1,1]).empty)

    def test_shared_color_normalization_and_export(self):
        f=pd.DataFrame({'x':[0], 'y':[0], 'depth':[5], 'value':[80]})
        with tempfile.TemporaryDirectory() as tmp, patch('module.observed_overlay.raw_observations',return_value=f):
            fig,ax=plt.subplots()
            image=ax.imshow([[0,50]],extent=[-1,1,-1,1],vmin=0,vmax=50,cmap='viridis_r')
            path=Path(tmp)/'points.csv'
            add_overlay(ax,image,'n',{'prediction_depth_m':5,'observed_depth_tolerance_m':1},path)
            self.assertIs(ax.collections[-1].norm,image.norm)
            self.assertEqual(pd.read_csv(path).value.tolist(),[80])
            plt.close(fig)

    def test_model_json_overrides_target_and_validates_plot_settings(self):
        def read(path):
            if path=='config/model.json': return {'prediction_depth_m':7,'observed_depth_tolerance_m':2}
            return {'prediction_depth_m':3}
        with patch('module.configuration._read_json',side_effect=read):
            c=load_prediction_settings('target.json')
            self.assertEqual((c['prediction_depth_m'],c['observed_depth_tolerance_m']),(7,2))
        for invalid in (-1, True, float('nan'), None):
            with patch('module.configuration._read_json',return_value={'prediction_depth_m':5,'observed_depth_tolerance_m':invalid}):
                with self.assertRaises(ValueError): load_prediction_settings('target.json')
