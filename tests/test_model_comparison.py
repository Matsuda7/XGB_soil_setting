import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.comparison_data import spatial_folds, load_target
from module.comparison_reporting import metrics
from module.comparison_models import predict_batches, build_model, check_capacity
from module.model_comparison import evaluate_target


class ComparisonTests(unittest.TestCase):
    def config(self):
        config = json.loads(Path('config/comparison.json').read_text())
        config['models'] = ['xgboost']
        config['xgboost'].update(n_estimators=4, n_jobs=1, max_depth=2)
        return config

    def data(self):
        return pd.DataFrame(dict(row_id=np.arange(40), source_row_id=np.arange(40),
            boring_id=np.repeat([f'h{i}' for i in range(20)], 2),
            x=np.repeat(np.arange(20) * 10000 + 100., 2), y=100., depth=5.,
            n_value=np.tile(np.arange(10) + 1., 4)))

    def test_spatial_groups_and_reproducibility(self):
        d = self.data()
        d.loc[2:3, 'boring_id'] = 'h0'  # A borehole spans two blocks.
        d.loc[4:5, 'x'] = d.loc[0, 'x']  # Another hole in the same block.
        a = spatial_folds(d, self.config())
        pd.testing.assert_frame_equal(a, spatial_folds(d, self.config()))
        self.assertEqual(a.groupby('boring_id').fold.nunique().max(), 1)
        self.assertEqual(a.groupby('block').fold.nunique().max(), 1)
        self.assertTrue(a.fold.between(1, 5).all())

    def test_too_few_groups_is_not_random_fallback(self):
        d = self.data(); d.x = 100
        with self.assertRaisesRegex(ValueError, 'independent spatial groups'):
            spatial_folds(d, self.config())

    def test_metrics_and_bias_sign(self):
        score = metrics(np.array([1., 2., 3.]), np.array([2., 3., 4.]))
        self.assertAlmostEqual(score['bias'], 1)
        self.assertAlmostEqual(score['rmse'], 1)
        self.assertAlmostEqual(score['mae'], 1)
        self.assertAlmostEqual(score['r2'], -.5)

    def test_capacity_never_subsamples(self):
        with patch('module.comparison_models.tabpfn_device', return_value='cpu'):
            with self.assertRaisesRegex(ValueError, 'No subsampling'):
                check_capacity(1001, 10, {})
            self.assertEqual(check_capacity(1000, 10, {}), 'cpu')
        with patch('module.comparison_models.tabpfn_device', return_value='cuda'):
            with self.assertRaises(ValueError):
                check_capacity(10001, 10, {})
            with self.assertRaises(ValueError):
                check_capacity(10, 501, {})

    def test_xgboost_does_not_need_tabpfn(self):
        with patch('module.comparison_models.version', side_effect=AssertionError('optional import')):
            model = build_model('xgboost', self.config(), 30, 3)
            self.assertEqual(model.get_params()['random_state'], 42)

    def test_batches_keep_order(self):
        class Model:
            def predict(self, values): return values[:, 0] * 2
        np.testing.assert_array_equal(predict_batches(Model(), np.arange(10).reshape(-1, 1), 3), np.arange(10)*2)

    def test_same_folds_preprocessing_and_saved_outputs(self):
        config = self.config(); config['models'] = ['xgboost', 'tabpfn']
        captured = []
        # An instrumented second estimator verifies orchestration only, not TabPFN accuracy.
        def builder(name, cfg, rows, columns):
            from module.xgb_common import build_regressor
            model = build_regressor(cfg['xgboost'])
            original = model.fit
            def fit(x, y):
                captured.append((name, x.copy(), y.copy()))
                return original(x, y)
            model.fit = fit
            return model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'data.csv'
            data = self.data(); data.loc[0, 'depth'] = 5
            data['slope'] = np.arange(40, dtype=float); data.loc[0, 'slope'] = np.nan
            data.to_csv(source, index=False)
            spec = dict(data=str(source), numeric_features=['x', 'depth', 'slope'], categorical_features=[])
            with patch('module.model_comparison.build_model', side_effect=builder), \
                 patch('module.model_comparison.save_model'):
                summary, folds, oof, errors = evaluate_target('N', spec, config, root)
            self.assertFalse(errors)
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(folds), 10)
            for i in range(0, len(captured), 2):
                np.testing.assert_array_equal(captured[i][1], captured[i+1][1])
                np.testing.assert_array_equal(captured[i][2], captured[i+1][2])
            for _, group in oof.groupby('model'):
                self.assertEqual(sorted(group.row_id), list(range(40)))
            for filename in ('oof_predictions.csv','metrics_by_fold.csv','metrics_summary.csv',
                             'scatter_comparison.png','residual_comparison.png','cv_comparison.png'):
                self.assertTrue((root / 'N' / filename).is_file())
            split = pd.read_csv(root/'N/cv_splits.csv')
            import joblib
            pre = joblib.load(root/'N/xgboost/fold_1/preprocessor.joblib')
            expected = data.loc[split.fold.ne(1), 'slope'].median()
            self.assertAlmostEqual(pre.named_transformers_['numeric'].named_steps['imputer'].statistics_[2], expected)

    def test_missing_tabpfn_is_incomplete_not_success(self):
        config=self.config();config['models']=['xgboost','tabpfn'];config['save_plots']=False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'data.csv';self.data().to_csv(source,index=False)
            spec=dict(data=str(source),numeric_features=['x','depth'],categorical_features=[])
            with patch('module.comparison_models.version', side_effect=RuntimeError('unavailable')):
                summary, folds, oof, errors=evaluate_target('N',spec,config,root)
            self.assertEqual([s['model'] for s in summary],['xgboost'])
            self.assertEqual(len(errors),1)

    def test_target_and_learned_features_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nested spatial'):
            load_target('c',dict(data='unused',numeric_features=['n_value'],categorical_features=[]),self.config())

    def test_distribution_shared_coordinates_and_raw_difference(self):
        from module.comparison_distribution import distribute
        config = self.config()
        config['models'] = ['xgboost', 'tabpfn']
        class Model:
            def __init__(self, value): self.value = value
            def fit(self, x, y): return self
            def predict(self, x): return np.full(len(x), self.value)
            def get_params(self): return {'constant': self.value}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data.csv'; self.data().to_csv(source, index=False)
            grid = root / 'grid.csv'
            pd.DataFrame(dict(x=[-45800., -45700.], y=[105400., 105400.], depth=[5., 5.])).to_csv(grid, index=False)
            config['targets'] = {'N': dict(data=str(source), numeric_features=['x', 'depth'], categorical_features=[])}
            config['distribution'].update(enabled=True, feature_csv=str(grid), chunk_size=1)
            with patch('module.comparison_distribution.build_model', side_effect=lambda n,*a: Model(-1 if n=='xgboost' else 2)), \
                 patch('module.comparison_distribution.save_model'):
                distribute(config, root)
            x = pd.read_csv(root/'N/xgboost/grid/000001.csv.gz')
            t = pd.read_csv(root/'N/tabpfn/grid/000001.csv.gz')
            delta = pd.read_csv(root/'N/difference/000001.csv.gz')
            pd.testing.assert_frame_equal(x[['x','y','depth']], t[['x','y','depth']])
            self.assertEqual(x.predicted_raw.iloc[0], -1)
            self.assertEqual(delta.difference_raw.iloc[0], 3)
            self.assertEqual(x.predicted_phi_deg.iloc[0], 15)
            self.assertTrue((root/'N/difference_distribution.png').exists())

    def test_comparison_cli_routes_without_legacy_execution(self):
        from module.pipeline import main
        with patch('module.model_comparison.run_comparison', return_value=0) as run:
            self.assertEqual(main(['--stage','compare','--models','tabpfn','--comparison-targets','c']),0)
            run.assert_called_once_with('config/comparison.json',['tabpfn'],['c'])


if __name__ == '__main__':
    unittest.main()
