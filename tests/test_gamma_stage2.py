"""Borehole exclusions must protect both teacher and student evaluation."""
import unittest
import pandas as pd
import numpy as np
from module.gamma_stage2 import exclude_heldout
from module.xgb_common import build_preprocessor


class Stage2LeakageTests(unittest.TestCase):
    def test_geology_onehot_is_train_only_and_accepts_unseen(self):
        train = pd.DataFrame({'x': [0., 1.], 'geology': ['A', 'B']})
        test = pd.DataFrame({'x': [2.], 'geology': ['UNSEEN']})
        pre = build_preprocessor(['x'], ['geology'])
        encoded = pre.fit_transform(train)
        np.testing.assert_array_equal(encoded[:, 1:], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(pre.transform(test)[:, 1:], [[0, 0]])
        self.assertNotIn('UNSEEN', ' '.join(pre.get_feature_names_out()))

    def test_excludes_holes_and_coordinates_but_keeps_nearby(self):
        held = pd.DataFrame({'boring_id': ['test'], 'x': [100.], 'y': [100.]})
        candidates = pd.DataFrame({
            'boring_id': ['test', 'duplicate_location', 'same_block', 'safe'],
            'x': [25000., 100., 9999., 10000.], 'y': [100., 100., 100., 100.]})
        safe = exclude_heldout(candidates, held)
        self.assertEqual(safe.boring_id.tolist(), ['same_block', 'safe'])

    def test_borehole_mode_still_excludes_duplicate_coordinates(self):
        held = pd.DataFrame({'boring_id': ['test'], 'x': [100.], 'y': [100.]})
        candidates = pd.DataFrame({'boring_id': ['other_id', 'nearby'], 'x': [100., 101.], 'y': [100., 100.]})
        safe = exclude_heldout(candidates, held)
        self.assertEqual(safe.boring_id.tolist(), ['nearby'])


if __name__ == '__main__':
    unittest.main()
