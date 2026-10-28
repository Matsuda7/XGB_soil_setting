import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from module.config_snapshot import write_config_snapshot


class ConfigSnapshotTests(unittest.TestCase):
    def test_all_targets_save_resolved_training_and_prediction_settings(self):
        from module.c_model_a import load_config as c_config
        from module.gamma_config import load_config as gamma_config
        from module.phi_prediction import load_prediction_config, DEFAULT_CONFIG
        from module import phi_training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c = c_config()
            configs={'c': c, 'gamma': gamma_config(),
                     'phi': load_prediction_config(DEFAULT_CONFIG, root/'phi')}
            required={'model_settings','numeric_features','categorical_features',
                      'validation','xgb_parameters','spatial','prediction_depth_m','output_dir','model_config'}
            for target, config in configs.items():
                before=deepcopy(config)
                write_config_snapshot(config,root/target,target)
                saved=json.loads((root/target/'config.json').read_text())
                self.assertTrue(required <= saved.keys(), target)
                self.assertEqual(config,before)
                self.assertTrue(Path(saved['spatial']['grid_input']).is_absolute())
                if target=='phi':
                    self.assertEqual(saved['xgb_parameters'],phi_training.XGB_PARAMETERS)
                    self.assertEqual(saved['validation'],phi_training.load_validation_config())

    def test_gamma_effective_n_feature_does_not_mutate_base_features(self):
        config={'numeric_features':['depth'], 'distribution_model':'predicted_n'}
        with tempfile.TemporaryDirectory() as tmp:
            saved=write_config_snapshot(config,Path(tmp),'gamma')
        self.assertEqual(saved['numeric_features'],['depth','n_value'])
        self.assertEqual(config['numeric_features'],['depth'])
