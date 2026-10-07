import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from module import phi_training as training
from module.xgb_common import regression_metrics


class PhiCrossValidationTests(unittest.TestCase):
    def test_real_five_fold_fit_and_saved_scores(self):
        # Repeated rows per hole detect leakage that row-wise KFold would miss.
        frame = pd.DataFrame({
            'boring_id': np.repeat([f'h{i}' for i in range(10)], 3),
            'measurement_no': np.tile([1, 2, 3], 10),
            'x': np.repeat(np.arange(10), 3), 'y': np.zeros(30),
            'depth': np.tile([1., 2., 3.], 10),
            'n_value': np.arange(30) % 20 + 1.,
        })
        config = dict(mode='borehole', n_splits=5, validation_size=.25, random_state=42)
        parameters = dict(training.XGB_PARAMETERS, n_estimators=4, max_depth=2,
                          n_jobs=1, early_stopping_rounds=2)
        with tempfile.TemporaryDirectory() as tmp, \
             training.output_paths(Path(tmp) / 'model'), \
             patch.object(training, 'XGB_PARAMETERS', parameters):
            _, _, oof, metrics = training.train_and_evaluate(frame, ['x', 'depth'], [], config)
            root = Path(tmp) / 'model'
            scores = pd.DataFrame(metrics['cross_validation']['folds'])
            unified = pd.read_csv(root.parent / 'evaluation_metrics.csv')
            self.assertEqual(unified.result.tolist(), [f'fold_{i}' for i in range(1, 6)] + ['fold_mean', 'fold_std', 'cv_oof'])
            self.assertFalse((root / 'cv_fold_metrics.csv').exists())
            splits = pd.read_csv(root / 'cv_splits.csv')
            self.assertEqual(scores.fold.tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(oof.cv_row_id.tolist(), list(range(30)))
            heldout = splits.loc[splits.split == 'test']
            self.assertTrue(heldout.groupby('cv_row_id').size().eq(1).all())
            self.assertTrue(heldout.groupby('boring_id').fold.nunique().eq(1).all())
            for fold in range(1, 6):
                part = splits.loc[splits.fold == fold]
                groups = [set(part.loc[part.split == name, 'boring_id'])
                          for name in ('train', 'validation', 'test')]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
                pred = pd.read_csv(root / f'evaluation/fold_{fold}/test_predictions.csv')
                expected = regression_metrics(pred.n_value, pred.predicted_n_value)
                for name, value in expected.items():
                    self.assertAlmostEqual(scores.set_index('fold').loc[fold, name], value)
                self.assertTrue((root / f'evaluation/fold_{fold}/xgb_spt_model.json').is_file())
            saved = metrics['cross_validation']
            self.assertEqual(saved['final_n_estimators'], int(scores.n_estimators.median()))
            for name in ('r2', 'rmse', 'mae'):
                self.assertAlmostEqual(saved['summary'][name]['mean'], scores[name].mean())
                self.assertAlmostEqual(saved['summary'][name]['std'], scores[name].std(ddof=1))
                self.assertAlmostEqual(metrics['test'][name], regression_metrics(oof.n_value, oof.predicted_n_value)[name])
            # Check the combined evaluation summary identifies the OOF population.
            from module.all_observed_reporting import save_comparison_summary
            metrics['all_observed'] = dict(r2=1., rmse=0., mae=0., rows=30,
                                          model_source='final refit', independent_test=False)
            (root / 'metrics.json').write_text(json.dumps(metrics))
            save_comparison_summary(root)
            combined = pd.read_csv(root / 'comparison_metrics.csv')
            self.assertEqual(combined.population.tolist(), ['cv_oof', 'all_observed'])

    def test_invalid_fold_settings_and_too_few_holes(self):
        data = pd.DataFrame({'boring_id': list('abcd')})
        for config in (dict(mode='row', n_splits=5), dict(mode='borehole', n_splits=4),
                       dict(mode='borehole', n_splits=True), dict(mode='borehole', n_splits=5)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                training.train_and_evaluate(data, ['depth'], [], config)
