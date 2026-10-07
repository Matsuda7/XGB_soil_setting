"""Selected gamma targets propagate from the CLI through single-target grids."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from module import pipeline, gamma_prediction
from module.gamma_config import selected_kinds


class GammaSelectionTests(unittest.TestCase):
    def test_config_rejects_multiple_or_unknown_targets(self):
        self.assertEqual(selected_kinds({}), ('wet',))
        for kind in ('wet', 'dry'):
            self.assertEqual(selected_kinds({'gamma_kind': kind}), (kind,))
        for kind in (None, ['wet', 'dry'], 'both', 'unknown'):
            with self.assertRaises(ValueError):
                selected_kinds({'gamma_kind': kind})

    def test_cli_passes_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.json'
            config.write_text('{"targets": ["gamma"]}')
            with patch.object(pipeline, 'run', return_value=0) as run:
                self.assertEqual(pipeline.main(['--config', str(config), '--gamma-kind', 'dry']), 0)
                run.assert_called_once_with('all', ('gamma',), str(config), False, gamma_kind='dry')

    def test_single_target_grid_outputs_only_selected_kind(self):
        for kind in ('wet', 'dry'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                points = root / 'points.txt'
                points.write_text('0 0 0 0\n1 0 0 0\n0 1 0 0\n1 1 0 0\n')
                config = {
                    'gamma_kind': kind, 'prediction_depth_m': 1.,
                    'input_chunk_size': 10, 'display_spacing_m': 1., 'write_text_matrix': True,
                    'spatial': {'property_mask_bounds': {'xmin': 0., 'xmax': 1., 'ymin': 0., 'ymax': 1.},
                                'grid_spacing_m': 1., 'property_mask_spacing_m': 1.,
                                'property_mask': 'unused', 'grid_input': str(points), 'grid_crs': 'EPSG:6675'},
                }
                models = {kind: (None, None, {'training_depth_range_m': [0., 2.]})}
                def predict(points, *args):
                    return points.assign(**{f'gamma_{kind}_kn_m3': 18.})
                with patch.object(gamma_prediction, 'read_binary_mask', return_value=np.ones((2, 2))), \
                     patch.object(gamma_prediction, 'load_dem'), \
                     patch.object(gamma_prediction, 'geographic_context', return_value=(None, None)), \
                     patch.object(gamma_prediction, 'predict_points', side_effect=predict), \
                     patch.object(gamma_prediction, 'plot_maps') as plot, \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = gamma_prediction.predict_grid(config, models, root / 'grid')
                self.assertEqual(result['status'], 'complete')
                self.assertIsNone(result['dry_above_wet_predictions'])
                self.assertEqual(list(plot.call_args.args[0]), [kind])
                self.assertTrue((root / f'grid/gamma_{kind}.txt.gz').exists())
                self.assertEqual([p.name for p in (root / 'grid').glob('*.npy')], [f'gamma_{kind}.npy'])
                chunk = pd.read_csv(root / 'grid/chunks/gamma_000001.csv.gz')
                self.assertEqual([c for c in chunk if c.startswith('gamma_')], [f'gamma_{kind}_kn_m3'])
