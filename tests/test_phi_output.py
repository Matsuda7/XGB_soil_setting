import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from module.phi_output import PLOTS, artifact_directory, organize_model_output


class PhiOutputTests(unittest.TestCase):
    def test_compact_layout_preserves_artifacts_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'model'
            root.mkdir()
            scores = {'r2': .7, 'rmse': 2., 'mae': 1.}
            cv = dict(folds=[dict(fold=i, **scores, test_rows=2, n_estimators=3) for i in range(1, 6)],
                      summary={k: dict(mean=v, std=0.) for k, v in scores.items()}, oof=scores)
            metrics = dict(cross_validation=cv, rows={'test': 10},
                           all_observed=dict(**scores, rows=10), training_min_depth_m=5., training_max_depth_m=20.)
            (root / 'metrics.json').write_text(json.dumps(metrics))
            (root / 'xgb_spt_model.json').write_text('saved model')
            for name in PLOTS - {'feature_importance.png'}:
                (root / name).write_bytes(b'plot')
            (root / 'feature_importance_grouped.png').write_bytes(b'importance')
            (root / 'evaluation').mkdir()
            (root / 'evaluation/keep.txt').write_text('fold details')
            self.assertEqual(artifact_directory(root), root)
            organize_model_output(root)
            organize_model_output(root)
            self.assertEqual({p.name for p in root.iterdir() if p.is_file()}, PLOTS)
            self.assertEqual(artifact_directory(root), root / 'artifacts')
            self.assertEqual((root / 'artifacts/xgb_spt_model.json').read_text(), 'saved model')
            self.assertEqual((root / 'artifacts/evaluation/keep.txt').read_text(), 'fold details')
            report = pd.read_csv(root.parent / 'evaluation_metrics.csv')
            self.assertEqual(report.result.tolist(), [f'fold_{i}' for i in range(1, 6)] + ['fold_mean', 'fold_std', 'cv_oof', 'all_observed'])
            self.assertTrue(report.training_min_depth_m.eq(5.).all())

    def test_saved_conditions_are_exported_without_reading_current_config(self):
        from module.phi_output import save_evaluation_table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'model'
            (root / 'artifacts').mkdir(parents=True)
            snapshot = dict(numeric_features=['depth'], categorical_features=['symbol'],
                            include_n50=True, training_max_depth_m=20, custom_option='saved')
            (root / 'artifacts/model_config.json').write_text(json.dumps(snapshot))
            metrics = dict(test=dict(r2=.5, rmse=2., mae=1.), excluded_observed_n_values=[50],
                           validation_config={'n_splits': 5}, xgb_parameters={'max_depth': 7},
                           best_iteration=99, final_fit_rows=123)
            save_evaluation_table(root, metrics)
            report = pd.read_csv(root.parent / 'evaluation_metrics.csv')
            row = report.iloc[0]
            self.assertEqual(json.loads(row.numeric_features), ['depth'])
            self.assertEqual(json.loads(row.categorical_features), ['symbol'])
            self.assertEqual(json.loads(row.model_settings_json)['custom_option'], 'saved')
            self.assertFalse(row.include_n50)
            self.assertEqual(row.training_max_depth_m, 20)
            self.assertEqual(json.loads(row.validation_config_json), {'n_splits': 5})
            self.assertEqual(json.loads(row.xgb_parameters_json), {'max_depth': 7})
            self.assertEqual(row.final_n_estimators, 100)
