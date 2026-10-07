import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.all_observed_reporting import save_all_observed


class AllObservedTests(unittest.TestCase):
    def test_all_rows_score_and_holdout_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'metrics.json').write_text(json.dumps({'test':{'r2':.2}, 'rows':{'test':1}}))
            (root/'observed_vs_predicted.png').write_bytes(b'original holdout')
            frame=pd.DataFrame({'boring_id':['a','b','c'], 'target':[1.,2.,3.]})
            report=save_all_observed(frame,[1.,2.,3.],'target','prediction',root,'observed_vs_predicted.png','test')
            self.assertEqual(report['rows'],3)
            self.assertEqual(report['r2'],1.)
            self.assertFalse(report['independent_test'])
            self.assertEqual((root/'observed_vs_predicted_test.png').read_bytes(),b'original holdout')
            metrics=json.loads((root/'metrics.json').read_text())
            summary=pd.read_csv(root/'comparison_metrics.csv')
            self.assertEqual(summary.population.tolist(), ['test','all_observed'])
            self.assertEqual(summary.r2.tolist(), [.2,1.])
            self.assertEqual(summary.rows.tolist(), [1,3])
            self.assertEqual(summary.independent_test.tolist(), [True,False])
            self.assertEqual(metrics['test']['r2'],.2)
            self.assertEqual(metrics['all_observed']['r2'],1.)
            self.assertEqual(len(pd.read_csv(root/'all_observed_predictions.csv')),3)
            save_all_observed(frame,[1.,1.,2.],'target','prediction',root,'observed_vs_predicted.png','test')
            self.assertEqual((root/'observed_vs_predicted_test.png').read_bytes(),b'original holdout')

    def test_cannot_drop_invalid_rows_or_include_pseudo_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=pd.DataFrame({'target':[1.,2.]})
            with self.assertRaises(ValueError):
                save_all_observed(frame,[1.,np.nan],'target','prediction',tmp,'plot.png','test')
            frame['label_source']=['measured','predicted']
            with self.assertRaises(ValueError):
                save_all_observed(frame,[1.,2.],'target','prediction',tmp,'plot.png','test')
