import unittest
import numpy as np
from src.analoggym_campaign import catalog, parameter_plan, observed_step, testbenches, ROOT
from src.analoggym_campaign_runner import design_fingerprint, metadata, normalize_deck


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.entries,cls.excluded=catalog(); cls.by_name={e['name']:e for e in cls.entries}

    def test_inventory_and_all_graph_terminals(self):
        self.assertEqual(len(self.entries),21)
        self.assertTrue(any(e['name']=='Qu_LEC_Pin_3' for e in self.excluded))
        for e in self.entries:
            g=e['graph']; expected=0
            for d in g['devices']:
                if d['model'] in {'nmos','pmos'}: self.assertEqual(d['ports'],['D','G','S','B'])
                expected+=len(d['ports'])*(len(d['ports'])-1)//2
            self.assertEqual(g['num_edges'],expected)
            self.assertEqual(g['num_nodes'],len(g['nets']))

    def test_parameter_aliases_match_and_determinism(self):
        e=self.by_name['ldo_simple']
        designs=[]
        for i in range(30):
            x=parameter_plan(e,i); self.assertEqual(x,parameter_plan(e,i))
            self.assertEqual(x['w_m1'],x['w_m2']); self.assertEqual(x['l_m3'],x['l_m4'])
            designs.append(design_fingerprint(e,x))
        self.assertEqual(len(set(designs)),30)

    def test_unstable_output_not_hidden_by_final_mean(self):
        t=np.linspace(0,1,10001); y=.3+.2*(1-np.exp(-t*20))+.04*np.sin(t*300)
        with self.assertRaises(ValueError): observed_step(t,y,0,1,.3,.5)
        stable=.3+.21*(1-np.exp(-t*30))
        sr,settle=observed_step(t,stable,0,1,.3,.51)
        self.assertGreater(sr,0); self.assertLess(settle,.2)

    def test_model_geometry_bounds_and_semantics(self):
        e=self.by_name['Tan_CLIA_Pin_3']; x=parameter_plan(e,0)
        self.assertGreaterEqual(x['mosfet_71_1_w_gm23_nmos'],.42)
        amp=metadata(self.by_name['Fan_SMC_Pin_3']); ldo=metadata(self.by_name['ldo_simple'])
        self.assertEqual(len(amp['metrics']),15); self.assertEqual(len(ldo['metrics']),13)
        self.assertNotEqual(amp['group_id'],ldo['group_id'])
        self.assertIn('observed',amp['metrics'][11]['name'])

    def test_fixture_comparison_never_ignores_load_or_temperature(self):
        a='Cload out 0 500p\n.temp 27\n.options reltol=1e-5\ntran 10n 401u 0 50n'
        b='cload out 0 5e-10\n.temp 27\n.options reltol=1e-7\ntran 10n 401u 0 25n'
        self.assertEqual(normalize_deck(a,True),normalize_deck(b,True))
        self.assertNotEqual(normalize_deck(a,True),normalize_deck(b.replace('5e-10','15n'),True))
        self.assertNotEqual(normalize_deck(a,True),normalize_deck(b.replace('.temp 27','.temp 25'),True))


if __name__=='__main__': unittest.main()
