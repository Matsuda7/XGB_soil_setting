import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from module import output_layout, distributions, phi_prediction, phi_training, spt_dataset


class OutputLayoutTests(unittest.TestCase):
    def test_date_uses_japan_time_and_session_keeps_same_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(output_layout, 'datetime') as clock:
                clock.now.return_value = datetime(2026, 9, 18, 0, 1)
                with output_layout.output_session(tmp) as directory:
                    self.assertEqual(directory, Path(tmp) / '20260918')
                    self.assertEqual(str(clock.now.call_args.args[0]), 'Asia/Tokyo')
                    clock.now.return_value = datetime(2026, 9, 19, 0, 1)
                    self.assertEqual(output_layout.current_directory(), directory)

    def test_same_day_target_rerun_preserves_old_results(self):
        with tempfile.TemporaryDirectory() as tmp, output_layout.output_session(tmp) as day:
            first = output_layout.create_target_directory('gamma_wet')
            (first / 'old.txt').write_text('old')
            dry = output_layout.create_target_directory('gamma_dry')
            (dry / 'dry.txt').write_text('dry')
            second = output_layout.create_target_directory('gamma_wet')
            self.assertEqual(second, first)
            self.assertEqual(list(second.iterdir()), [])
            self.assertEqual(next((day / 'archive').glob('gamma_wet_*/old.txt')).read_text(), 'old')
            self.assertEqual((dry / 'dry.txt').read_text(), 'dry')

    def test_phi_training_and_prediction_share_new_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp, output_layout.output_session(tmp) as day:
            def train(directory):
                with phi_training.output_paths(directory):
                    self.assertEqual(phi_training.output_path('metrics.json'), day / 'phi/model/metrics.json')
            with patch.object(phi_training, 'main', side_effect=train), \
                 patch.object(phi_prediction, 'main') as predict:
                result = distributions.distribute('phi')
            self.assertEqual(result['output'], str(day / 'phi'))
            predict.assert_called_once_with([], output_dir=day / 'phi')
            config = json.loads((day / 'phi/config.json').read_text())
            self.assertEqual(config['phi_text'], str(day / 'phi/grid/phi_10m.txt.gz'))
            self.assertFalse(any('bedrock' in key for key in config))
            self.assertEqual(json.loads((day / 'phi/run_summary.json').read_text())['status'], 'complete')

    def test_gamma_config_selection_is_used_by_dispatch(self):
        for kind in ('wet', 'dry'):
            with patch('module.gamma_config.load_config', return_value={'gamma_kind': kind}), \
                 patch('module.gamma.distribute', return_value={'status': 'complete'}) as run:
                distributions.distribute('gamma')
                run.assert_called_once_with({'gamma_kind': kind})

    def test_phi_dataset_save_does_not_require_bedrock_columns(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'model_dataset.csv'
            spt_dataset.save_dataset(pd.DataFrame({'boring_id': ['a'], 'n_value': [5]}), target)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['model_dataset.csv'])

    def test_phi_display_reads_prediction_without_bedrock(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'x': [0., 10.], 'y': [0., 10.], 'predicted_n_value': [5., 10.]}).to_csv(
                root / 'N_value_10m_000001.csv.gz', index=False)
            result = phi_prediction.load_display_points(root, 10.)
            self.assertEqual(list(result.columns), ['x', 'y', 'predicted_n_value'])
            self.assertEqual(len(result), 2)

    def test_gamma_execution_writes_selected_target_inside_date(self):
        from module import gamma
        for kind in ('wet', 'dry'):
            with tempfile.TemporaryDirectory() as tmp, output_layout.output_session(tmp) as day:
                root = Path(tmp)
                training = root / 'training/gamma'
                training.mkdir(parents=True)
                (training / 'preparation_summary.json').write_text('{"signature": {}}')
                config = {'gamma_kind': kind, 'prediction_depth_m': 1.,
                          'n_comparison': {'enabled': False}, 'create_distribution': True,
                          'distribution_model': 'no_n'}
                with patch('module.paths.TRAINING', root / 'training'), \
                     patch('module.gamma_config.data_signature', return_value={}), \
                     patch.object(gamma, 'train_both', return_value={kind: 'model'}), \
                     patch.object(gamma, 'predict_grid', return_value={'status': 'complete'}) as predict:
                    result = gamma.distribute(config)
                output = day / ('gamma_' + kind)
                self.assertEqual(result['output'], str(output))
                self.assertEqual(predict.call_args.args[2], output / 'grid')
                self.assertEqual(json.loads((output / 'run_summary.json').read_text())['gamma_kind'], kind)
                self.assertFalse((day / 'gamma').exists())
