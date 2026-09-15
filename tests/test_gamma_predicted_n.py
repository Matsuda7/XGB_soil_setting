"""Grid coupling checks, added without executing tests or any analysis."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.gamma_predicted_n import add_predicted_n
from module.gamma_prediction import predict_points


class IdentityTransform:
    def transform(self, frame):
        return frame


class EchoNModel:
    def predict(self, frame):
        return frame.n_input.to_numpy()


class PredictedNTests(unittest.TestCase):
    def test_depth_elevation_and_ensemble_mean(self):
        frame=pd.DataFrame({'x':[1.], 'y':[2.], 'surface_z':[12.], 'depth':[3.]})
        metrics={'numeric_features':['n_value_elevation'], 'categorical_features':[], 'interval_radius':4.}
        def predict(models, transform, query, numeric, categorical):
            self.assertEqual(query.n_value_elevation.iloc[0],9.)
            return np.array([[10.],[20.],[30.]])
        with patch('module.gamma_predicted_n.predict_ensemble',side_effect=predict):
            result=add_predicted_n(frame,([],None,metrics))
        self.assertEqual(result.n_input.iloc[0],20.)
        self.assertEqual(result.n_lower.iloc[0],16.)
        self.assertEqual(result.n_upper.iloc[0],24.)

    def test_grid_passes_predicted_n_to_both_gamma_models(self):
        config={'distribution_model':'predicted_n','gravity_m_s2':9.80665}
        points=pd.DataFrame({'x':[1.], 'y':[2.]})
        def enrich(frame,*args):
            return frame.assign(surface_z=10.)
        def add(frame,artifact):
            self.assertEqual(frame.depth.iloc[0],2.)
            return frame.assign(n_input=25.,n_ensemble_std=1.,n_lower=20.,n_upper=30.)
        metrics={'numeric_features':['n_input'],'categorical_features':[]}
        models={kind:(EchoNModel(),IdentityTransform(),metrics) for kind in ('wet','dry')}
        with patch('module.gamma_prediction.enrich',side_effect=enrich), patch('module.gamma_predicted_n.add_predicted_n',side_effect=add):
            result=predict_points(points,2.,config,(None,None,None),models,n_artifact=object())
        self.assertEqual(result.gamma_wet_kn_m3.iloc[0],25.)
        self.assertEqual(result.gamma_dry_kn_m3.iloc[0],25.)
        self.assertAlmostEqual(result.density_wet_g_cm3.iloc[0],25./9.80665)

    def test_missing_upstream_model_fails(self):
        with patch('module.gamma_prediction.enrich',return_value=pd.DataFrame({'x':[1.],'y':[2.]})):
            with self.assertRaisesRegex(ValueError,'Missing N model'):
                predict_points(pd.DataFrame({'x':[1.],'y':[2.]}),1.,{'distribution_model':'predicted_n'},(None,None,None),{})
