"""Model A target semantics; not executed during source-only implementation."""
import unittest
import numpy as np
import pandas as pd
from module.c_model_a import select_targets


class ModelATargetTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'test_condition_code':['B0523','B0524','B0522','B0521','B0524','B0523'],
            'test_type':['CUb','CD','CU','UU','CD','CUb'],
            'c_effective':[np.nan]*6,
            'shear_strength_effective':[0.,12.,20.,30.,np.nan,-2.],
            'shear_strength_total':[100.]*6,
            'depth_mid':[1.]*6,
        })

    def test_only_effective_cub_cd_and_zero_are_accepted(self):
        accepted,rejected=select_targets(self.frame())
        self.assertEqual(accepted.target.tolist(),[0.,12.])
        self.assertEqual(len(rejected),4)
        self.assertEqual(set(accepted.target_unit),{'kPa'})

    def test_no_total_stress_substitution(self):
        frame=self.frame().iloc[[4]]
        accepted,rejected=select_targets(frame)
        self.assertTrue(accepted.empty)
        self.assertEqual(rejected.exclusion_reason.iloc[0],'missing_or_invalid_effective_cohesion')

    def test_conflicting_fields_are_rejected(self):
        frame=self.frame().iloc[[1]].copy();frame['c_effective']=18.
        accepted,rejected=select_targets(frame)
        self.assertTrue(accepted.empty)
        self.assertEqual(rejected.exclusion_reason.iloc[0],'conflicting_effective_cohesion_fields')
