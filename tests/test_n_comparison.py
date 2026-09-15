"""Future checks for matching and leakage exclusion. Not executed during implementation."""
import unittest
import numpy as np
import pandas as pd
from module.n_matching import match_measurements, location_groups
from module.n_estimation import exclude_queries


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.options={'xy_tolerance_m':.01,'depth_tolerance_m':1e-6}
        self.density=pd.DataFrame(dict(boring_id=['xml-a'],x=[100.],y=[200.],
                                      depth=[2.5],depth_top=[2.],depth_bottom=[3.]))
        self.spt=pd.DataFrame(dict(boring_id=['spt-a']*3,x=[100.]*3,y=[200.]*3,
                                  depth=[2.,3.,4.],n_value=[10.,30.,50.],measurement_no=[1,2,3]))

    def test_interval_mean_preserves_sources(self):
        row=match_measurements(self.density,self.spt,self.options).iloc[0]
        self.assertEqual(row.n_measured,20.)
        self.assertEqual(row.spt_count,2)
        self.assertEqual(row.n_match_method,'interval_mean')
        self.assertEqual(row.spt_max_depth_difference_m,.5)

    def test_nearest_is_not_implicitly_measured(self):
        sample=self.density.assign(depth=5.5,depth_top=5.,depth_bottom=6.)
        row=match_measurements(sample,self.spt,self.options).iloc[0]
        self.assertTrue(np.isnan(row.n_measured))
        self.assertEqual(row.nearest_spt_depth_m,4.)
        self.assertEqual(row.n_match_method,'no_spt_in_interval')

    def test_ambiguous_xy_is_not_arbitrarily_selected(self):
        spt=pd.concat([self.spt,self.spt.assign(boring_id='other-hole')],ignore_index=True)
        row=match_measurements(self.density,spt,self.options).iloc[0]
        self.assertEqual(row.n_match_method,'ambiguous_hole')
        self.assertTrue(np.isnan(row.n_measured))

    def test_coincident_xml_names_share_split_identity(self):
        frame=pd.concat([self.density,self.density.assign(boring_id='xml-b',x=100.005)])
        self.assertEqual(location_groups(frame,.01).nunique(),1)


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.spt=pd.DataFrame(dict(boring_id=['different-name','different-name','nearby','far'],
                                  x=[100.,100.,200.,2000.],y=[100.]*4,depth=[1.,10.,2.,2.]))
        self.heldout=pd.DataFrame(dict(boring_id=['xml-test'],x=[100.],y=[100.]))

    def test_borehole_exclusion_removes_every_depth_despite_name_difference(self):
        kept=exclude_queries(self.spt,self.heldout,{'mode':'borehole'},.01)
        self.assertEqual(set(kept.boring_id),{'nearby','far'})

    def test_spatial_exclusion_removes_all_holes_in_test_block(self):
        kept=exclude_queries(self.spt,self.heldout,{'mode':'spatial','block_size_m':1000.},.01)
        self.assertEqual(set(kept.boring_id),{'far'})
