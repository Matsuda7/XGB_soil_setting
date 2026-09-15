"""Validation split invariants; added without executing tests or training."""
import unittest
import numpy as np
import pandas as pd
from module.validation import split_data


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(mode='spatial', additional_spatial=True,
                           block_size_m=1000., test_size=.2,
                           validation_size=.2, random_state=42)
        # Two holes share each spatial block; three depths per hole.
        holes = np.repeat(np.arange(60), 3)
        self.data = pd.DataFrame(dict(boring_id=holes.astype(str),
            x=(holes // 2)*1000.+10., y=np.zeros(len(holes)),
            depth=np.tile([1.,2.,3.], 60), n_value=np.ones(len(holes))))

    def test_spatial_blocks_and_all_depths_are_disjoint(self):
        parts = split_data(self.data, self.config)
        self.assertEqual(sum(map(len, parts)), len(self.data))
        for column in ('boring_id', '_validation_group'):
            groups = [set(p[column]) for p in parts]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        blocks = [set((p.x // 1000).astype(int)) for p in parts]
        self.assertFalse(blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])

    def test_borehole_mode_works_without_spatial_blocks(self):
        data = self.data.assign(x=0.)
        parts = split_data(data, dict(self.config, mode='borehole', additional_spatial=False))
        groups = [set(p.boring_id) for p in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_invalid_or_insufficient_spatial_groups_fail(self):
        with self.assertRaises(ValueError):
            split_data(self.data, dict(self.config, block_size_m=0))
        with self.assertRaises(ValueError):
            split_data(self.data.assign(x=0.), self.config)

    def test_row_mode_can_disable_grouping(self):
        parts = split_data(self.data, dict(self.config, mode='row', additional_spatial=False))
        self.assertEqual(sum(map(len, parts)), len(self.data))
        self.assertTrue(set(parts[0].boring_id) & set(parts[2].boring_id))
