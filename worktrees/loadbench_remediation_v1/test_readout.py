import unittest
import numpy as np
from readout import predict, normalize
from supervision import supervision_view,measured_unity_crossings


class ReadoutChecks(unittest.TestCase):
    def test_quality_is_not_missingness(self):
        original=np.array([[True,True,True,False]])
        out=supervision_view(original,['ft','phasemargin','p','area'],np.array([True]))
        np.testing.assert_array_equal(original,out['observed_mask'])
        self.assertEqual(out['engineering_eligible'].tolist(),[[False,False,True,False]])

    def test_extractor_cannot_fabricate_crossing(self):
        r=measured_unity_crossings([1,10,100],[-5,-6,-20])
        self.assertEqual(r['crossings'],[])
        r=measured_unity_crossings([1,10,100],[20,0,-20],[0,-1,-2])
        self.assertAlmostEqual(r['crossings'][0]['frequency_Hz'],10.)
        self.assertIsNone(r['crossings'][0]['phase_margin_degree'])
        self.assertFalse(r['extrapolation_used'])

    def test_affine_and_missing_neighbors(self):
        rng=np.random.default_rng(42); x=rng.normal(size=(80,3)); q=x[:7]+.001
        y=np.column_stack([2+x[:,0]-.4*x[:,1], np.full(80,7.)])
        mask=np.ones_like(y,dtype=bool);mask[:40,1]=False
        pred,n=predict(x,q,y,mask,ridge=1e-8)
        self.assertLess(np.max(abs(pred[:,0]-(2+q[:,0]-.4*q[:,1]))),1e-5)
        np.testing.assert_allclose(pred[:,1],7.,atol=1e-10)
        np.testing.assert_array_equal(n,[80,40])

    def test_train_only_stats_and_duplicates(self):
        x=np.array([[1.,2.],[1.,2.],[3.,2.]])
        a,b,mu,sd,active=normalize(x,np.array([[100.,-9.]]))
        np.testing.assert_allclose(mu,x.mean(0));self.assertEqual(active.tolist(),[True,False])
        z=np.array([[1.],[2.],[3.]])
        p,_=predict(a,b,z,np.ones_like(z,dtype=bool))
        self.assertTrue(1<=p[0,0]<=3)

    def test_absent_labels_do_not_change_other_task(self):
        x=np.arange(12.)[:,None];y=np.column_stack([x[:,0],x[:,0]+100])
        mask=np.ones_like(y,dtype=bool); mask[:6,1]=False
        a,_=predict(x,x[:3],y,mask)
        y[:6,1]=-1e30
        b,_=predict(x,x[:3],y,mask)
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
