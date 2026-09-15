import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from module.test_locations import read_binary_mask, boundary_segments, build_sites

class LocationTests(unittest.TestCase):
    def test_nonzero_includes_negative_and_large_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prop.txt'
            path.write_text('0 -2 10\n0 0 1\n')
            np.testing.assert_array_equal(read_binary_mask(path,(2,3)), [[0,1,1],[0,0,1]])
            with self.assertRaises(ValueError): read_binary_mask(path,(3,3))

    def test_boundary_preserves_position_across_strips(self):
        mask=np.array([[0,0,1,1]]*5,dtype=np.uint8)
        segments=boundary_segments(mask,{'xmin':100,'ymin':200},10,block_rows=2)
        self.assertEqual(len(segments),2)
        for segment in segments: np.testing.assert_allclose(segment[:,0],115)
        values=np.concatenate(segments)
        self.assertEqual(values[:,1].min(),200)
        self.assertEqual(values[:,1].max(),240)
        self.assertEqual(boundary_segments(np.ones((3,3)),{'xmin':0,'ymin':0},10),[])

    def test_sites_do_not_treat_shear_strength_as_cohesion(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'one.xml').write_text('<root><測地系>02</測地系></root>',encoding='shift_jis')
            frame=pd.DataFrame({'boring_id':['a','a'],'xml_file':['one.xml']*2,
                'longitude':[137.,137.], 'latitude':[37.,37.],
                'shear_strength_total':[10,20], 'wet_density':[1.8,np.nan]})
            sites,excluded=build_sites(frame,tmp,'EPSG:6675')
            self.assertEqual(len(sites),1)
            self.assertEqual(int(sites.records.iloc[0]),2)
            self.assertTrue(sites.c_related.iloc[0])
            self.assertFalse(sites.has_c.iloc[0])
            self.assertTrue(sites.wet.iloc[0])
            self.assertEqual(len(excluded),0)

class RawDensityTests(unittest.TestCase):
    def test_density_only_record_is_included(self):
        from module.test_locations import load_test_records
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'density.xml').write_text('<root><緯度>37</緯度><経度>137</経度><測地系>02</測地系><試験情報><試料番号>S1</試料番号><湿潤密度>1.8</湿潤密度></試験情報></root>', encoding='shift_jis')
            frame=load_test_records(tmp)
            self.assertEqual(len(frame),1)
            self.assertEqual(frame.wet_density.iloc[0],1.8)
            sites,_=build_sites(frame,tmp,'EPSG:6675')
            self.assertTrue(sites.wet.iloc[0])
            self.assertFalse(sites.c_related.iloc[0])
