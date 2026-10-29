import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from module import full_distribution as full


class FullDistributionTests(unittest.TestCase):
    def test_lock_and_matrix_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with full.job_lock(p/'job.lock'):
                with self.assertRaisesRegex(RuntimeError,'Another writer'):
                    with full.job_lock(p/'job.lock'):pass
            a=full.matrix(p/'grid.npy',(2,3));a[0,1]=12.;a.flush()
            b=full.matrix(p/'grid.npy',(2,3))
            self.assertEqual(b[0,1],12)
            self.assertTrue(np.isnan(b[1,1]))
            with self.assertRaises(ValueError):full.matrix(p/'grid.npy',(3,2))

    def test_cached_grid_inference_uses_dense_like_cv(self):
        # Sparse explicit zeros must not become XGBoost missing values.
        class Pre:
            def transform(self,frame):return sparse.csr_matrix([[0.,1.],[1.,0.]])
        class Model:
            def predict(self,x):
                self_test.assertIsInstance(x,np.ndarray)
                self_test.assertEqual(x.dtype,np.float32)
                return np.array([1.,2.])
        self_test=self
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'features').mkdir();(r/'c/xgboost').mkdir(parents=True)
            path=r/'features/000001.csv.gz';pd.DataFrame({'x':[0.,10.],'y':[0.,0.],'depth':[5.,5.]}).to_csv(path,index=False)
            manifest={'rows':2,'chunks':[{'file':path.name,'sha256':full.sha256(path),'rows':2}]}
            (r/'feature_manifest.json').write_text(json.dumps(manifest))
            cfg={'targets':{'c':{'numeric_features':['x'],'categorical_features':[]}}}
            spatial={'grid_spacing_m':10.,'property_mask_spacing_m':10.,'property_mask_bounds':{'xmin':0.,'xmax':10.,'ymin':0.,'ymax':10.}}
            with patch.object(full,'fit_or_load',return_value=(Model(),Pre())),patch.object(full,'plot_matrix'):
                full.recompute_xgboost(cfg,spatial,r,r)
            a=np.load(r/'c/xgboost/raw.npy')
            np.testing.assert_array_equal(a[0],[1.,2.])
            self.assertTrue(np.isnan(a[1]).all())
            self.assertEqual(json.loads((r/'prepare_status.json').read_text())['status'],'complete')

    def test_tabpfn_stop_preserves_resume_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'features').mkdir();(r/'c/tabpfn').mkdir(parents=True)
            path=r/'features/000001.csv.gz';pd.DataFrame({'x':[0.,10.],'y':[0.,0.],'depth':[5.,5.]}).to_csv(path,index=False)
            (r/'feature_manifest.json').write_text(json.dumps({'rows':2,'chunks':[{'file':path.name,'sha256':full.sha256(path),'rows':2}]}))
            (r/'STOP_TABPFN').touch()
            cfg={'tabpfn':{},'eligible_models':{'c':['tabpfn']},'targets':{'c':{'numeric_features':['x'],'categorical_features':[]}}}
            spatial={'grid_spacing_m':10.,'property_mask_spacing_m':10.,'property_mask_bounds':{'xmin':0.,'xmax':10.,'ymin':0.,'ymax':10.}}
            class Pre:
                def transform(self,x):return x.to_numpy()
            with patch.object(full,'fit_or_load',return_value=(None,Pre())):
                full.tabpfn(cfg,spatial,r,r,'low_memory',1)
            s=json.loads((r/'c/tabpfn/prediction_status.json').read_text())
            self.assertEqual((s['status'],s['rows'],s['offset']),('stopped',0,0))
            self.assertFalse((r/'c/tabpfn/distribution.png').exists())


if __name__=='__main__':unittest.main()
