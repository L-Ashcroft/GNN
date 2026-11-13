import unittest
import numpy as np
import pandas as pd
from src.stratified_sampling import sample_parameters,hierarchical_quotas,parameter_strata
from src.load_loadbench import quality_mask,definitions,resolve,csv_to_si,graph_from_combined


class FusionTests(unittest.TestCase):
    def test_real_sentinel_is_not_missing(self):
        f=pd.DataFrame({'dc_gain':[40.,40.,40.],'ft':[1e6]*3,'p':[1e-4]*3,
                        'sr_pos':[1e7,-1e13,np.nan],'sr_neg':[-1e7]*3})
        self.assertEqual(quality_mask(f).tolist(),[True,False,False])

    def test_matching_aliases_and_units(self):
        defs=definitions('.param param_w1=10u\n.param param_w2=param_w1\n.param param_m1=1')
        v=resolve(defs,{'param_w1':csv_to_si('param_w1',20.)})
        self.assertEqual(v['param_w1'],v['param_w2']);self.assertAlmostEqual(v['param_w2'],20e-6)
        self.assertEqual(csv_to_si('param_cc1',2e-12),2e-12)

    def test_zero_compensation_cap_is_not_negative_geometry(self):
        self.assertEqual(resolve({'param_cc1':'0'}, {})['param_cc1'],0.)
        with self.assertRaises(ValueError):resolve({'param_w1':'0'}, {})
        f=pd.DataFrame({'dc_gain':[40.],'ft':[1e6],'p':[1e-4], 'param_cc1':[0.]})
        self.assertTrue(quality_mask(f)[0])

    def test_bulk_tie_and_separate_device_types(self):
        text='.subckt op ib ip im out vss vdd\nX1 out ip vss vss sky130_fd_pr__nfet_01v8 w={param_w1} l={param_l1} m=1 nf=1\nX2 out im vdd vdd sky130_fd_pr__pfet_01v8 w={param_w2} l={param_l2}\n.ends op\nVdd vdd 0 1.8'
        graph,_=graph_from_combined(text)
        self.assertEqual(graph['num_edges'],12)
        self.assertEqual([d['model'] for d in graph['devices']],['nmos','pmos'])
        self.assertEqual(graph['devices'][0]['ports'],['D','G','S','B'])

    def test_distribution_sampling_deterministic_and_label_independent(self):
        rng=np.random.default_rng(3);p=rng.uniform(.1,2,(1000,4))
        f=pd.DataFrame({'parameters':p.tolist(),'performance':rng.random(1000)})
        a,stats=sample_parameters(f,200,'topology');f.performance=999
        b,_=sample_parameters(f,200,'topology')
        self.assertEqual(a.index.tolist(),b.index.tolist());self.assertEqual(len(set(a.index)),200)
        self.assertEqual(sum(s['selected'] for s in stats),200)
        self.assertTrue(all(s['selected']>0 for s in stats))
        np.testing.assert_array_equal(parameter_strata(p,'topology'),parameter_strata(p*1e6,'topology'))

    def test_topology_diversity_with_small_pool(self):
        specs=[{'topology_id':'a','circuit_family':'amp','connectivity_group':'g1'},
               {'topology_id':'b','circuit_family':'amp','connectivity_group':'g2'},
               {'topology_id':'c','circuit_family':'vco','connectivity_group':'g3'}]
        q=hierarchical_quotas(specs,{'a':2,'b':100,'c':100},50)
        self.assertEqual(sum(q.values()),50);self.assertEqual(q['a'],2);self.assertGreater(q['b'],0);self.assertGreater(q['c'],0)
