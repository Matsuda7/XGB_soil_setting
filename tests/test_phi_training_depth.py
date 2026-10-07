import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from module import phi_training as training
from module.xgb_common import load_jsonc_config, load_model_config


class PhiTrainingDepthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'data.csv'
        self.frame = pd.DataFrame({
            'boring_id': list('abcdefg'), 'x': range(7), 'y': range(7),
            'depth': [0., 10., 20., 20.01, np.nan, -1., np.inf],
            'n_value': [1.] * 7,
        })
        self.frame.to_csv(self.path, index=False)

    def test_limit_inclusive_even_without_depth_as_feature(self):
        data = training.load_dataset(self.path, ['x'], [], 20.)
        self.assertEqual(data.boring_id.tolist(), ['a', 'b', 'c'])
        self.assertEqual(data.attrs['training_depth_filter']['excluded_rows'], 4)
        self.assertEqual(len(pd.read_csv(self.path)), 7)
        self.assertEqual(len(training.load_dataset(self.path, ['x'], [], 0)), 1)

    def test_unlimited_preserves_previous_population(self):
        self.assertEqual(len(training.load_dataset(self.path, ['x'], [])), 7)
        self.assertEqual(len(training.load_dataset(self.path, ['x'], [], None)), 7)

    def test_invalid_limits_and_empty_selection(self):
        for maximum in (-1, np.inf, np.nan, True, '20', []):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, 'training_max_depth_m'):
                training.load_dataset(self.path, ['x'], [], maximum)
        self.frame.assign(depth=30.).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'No training records'):
            training.load_dataset(self.path, ['x'], [], 20)
        self.frame.drop(columns='depth').to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'depth column'):
            training.load_dataset(self.path, ['x'], [], 20)

    def test_minimum_and_maximum_are_inclusive(self):
        data = training.load_dataset(self.path, ['x'], [], 20., 10.)
        self.assertEqual(data.boring_id.tolist(), ['b', 'c'])
        self.assertEqual(data.attrs['training_depth_filter']['training_min_depth_m'], 10.)
        data = training.load_dataset(self.path, ['x'], [], None, 20.)
        self.assertEqual(data.boring_id.tolist(), ['c', 'd'])
        data = training.load_dataset(self.path, ['x'], [], 20., 20.)
        self.assertEqual(data.boring_id.tolist(), ['c'])
        for minimum in (-1, True, np.inf, np.nan, '10'):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, 'training_min_depth_m'):
                training.load_dataset(self.path, ['x'], [], None, minimum)
        with self.assertRaisesRegex(ValueError, 'must not exceed'):
            training.load_dataset(self.path, ['x'], [], 10., 20.)

    def test_include_n50_switch_and_validation(self):
        self.frame['n_value'] = [50., 10., 50., 20., 1., 1., 1.]
        self.frame.to_csv(self.path, index=False)
        included = training.load_dataset(self.path, ['x'], [], 20.)
        excluded = training.load_dataset(self.path, ['x'], [], 20., None, False)
        self.assertEqual(included.boring_id.tolist(), ['a', 'b', 'c'])
        self.assertEqual(excluded.boring_id.tolist(), ['b'])
        self.assertEqual(excluded.attrs['training_n_filter']['excluded_rows'], 2)
        self.assertEqual(excluded.attrs['training_depth_filter']['retained_rows'], 3)
        self.assertEqual(len(pd.read_csv(self.path)), 7)
        for invalid in (None, 0, 1, 'false'):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'include_n50'):
                training.load_dataset(self.path, ['x'], [], include_n50=invalid)
        self.frame['n_value'] = 50.
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'No training records'):
            training.load_dataset(self.path, ['x'], [], include_n50=False)

    def test_jsonc_option_works_with_feature_loader(self):
        config = self.root / 'model.jsonc'
        config.write_text('{\n// depth limit\n"training_max_depth_m": 20,\n'
                          '"numeric_features": ["x",], "categorical_features": [],\n}')
        self.assertEqual(load_jsonc_config(config)['training_max_depth_m'], 20)
        self.assertEqual(load_model_config(config), (['x'], []))

    def test_training_uses_same_filtered_population_and_saves_limit(self):
        config = self.root / 'model.jsonc'
        config.write_text(json.dumps({'numeric_features': ['x'], 'categorical_features': [],
                                      'training_max_depth_m': 20., 'training_min_depth_m': 10., 'include_n50': False}))
        from unittest.mock import Mock
        model, preprocessor = Mock(), Mock()
        model.predict.return_value = np.ones(2)
        with patch.object(training, 'MODEL_CONFIG', config), \
             patch.object(training, 'INPUT_CSV', self.path), \
             patch.object(training, 'train_and_evaluate', return_value=(None, None, None, {'best_iteration': 1})) as evaluate, \
             patch.object(training, 'fit_final_model', return_value=(model, preprocessor)) as fit, \
             patch.object(training, 'save_artifacts') as save, \
             patch.object(training, 'print_summary'), \
             patch('module.all_observed_reporting.save_all_observed', return_value={}) as report:
            training.main(self.root / 'model')
        for call in (evaluate, fit, report):
            self.assertEqual(call.call_args.args[0].boring_id.tolist(), ['b', 'c'])
        metrics = save.call_args.args[3]
        self.assertEqual(metrics['training_max_depth_m'], 20.)
        self.assertEqual(metrics['training_min_depth_m'], 10.)
        self.assertFalse(metrics['include_n50'])
        self.assertEqual(metrics['final_fit_rows'], 2)
        metadata = json.loads((self.root / 'model/artifacts/training_depth_filter.json').read_text())
        self.assertEqual(metadata['retained_rows'], 2)
        self.assertEqual(metadata['training_min_depth_m'], 10.)
        self.assertEqual(metadata['input_rows'], 7)
        self.assertEqual(json.loads((self.root / 'model/artifacts/model_config.json').read_text())['training_max_depth_m'], 20.)
