import copy
from pathlib import Path
import unittest
from src.common import read
from src.analoggym_campaign import parameter_plan as original
from src.analoggym_personal_engine import parameter_plan,sampling_trace


class FoldedSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('data/generated/analoggym_folded_optimization_v2/plan.json')
        if not path.exists():raise unittest.SkipTest('No optimizer plan')
        cls.plan=read(path)

    def entry(self,mode):
        e=copy.deepcopy(self.plan['entry'])
        e['_sampling_policy']={**self.plan['strategies'][mode],'start_candidate':100,'seed_offset':20261028,
                              'version':'test_v2','vary_discrete':False}
        return e

    def test_old_candidates_exactly_unchanged(self):
        e=self.entry('narrow_recentered')
        for c in [0,1,5,42,99]:self.assertEqual(parameter_plan(e,c),original(e,c))
        self.assertEqual(sampling_trace(e,99)['sampling_policy_version'],'personal_sampler_v1')

    def test_correlated_internal_ratios_and_aliases(self):
        e=self.entry('correlated_recentered')
        for c in range(100,110):
            p=parameter_plan(e,c);centre=e['_sampling_policy']['centers'][(c-100)%len(e['_sampling_policy']['centers'])]
            self.assertEqual(p['w_m1'],p['w_m2']);self.assertEqual(p['l_m7'],p['l_m8'])
            self.assertAlmostEqual((p['w_m7']/p['l_m7'])/(p['w_m9']/p['l_m9']),
                                   (centre['w_m7']/centre['l_m7'])/(centre['w_m9']/centre['l_m9']),places=12)
            for key in p:
                if key.startswith('m_'):self.assertEqual(p[key],centre[key])
            self.assertEqual(p,parameter_plan(e,c))

    def test_narrow_bounds_and_bias(self):
        e=self.entry('narrow_recentered')
        for c in range(100,110):
            p=parameter_plan(e,c);centre=e['_sampling_policy']['centers'][(c-100)%3]
            for k,v in p.items():
                if k.startswith(('w_','l_')):self.assertTrue(1/1.05-1e-12<=v/centre[k]<=1.05+1e-12)
                if k.startswith('vb'):self.assertTrue(1/1.02-1e-12<=v/centre[k]<=1.02+1e-12)

    def test_other_topologies_exactly_unchanged(self):
        recipe=read(Path('data/generated/analoggym_personal_20k_v1/recipe.json'))
        for entry in recipe['entries']:
            if entry['name']=='ldo_folded_cascode':continue
            e=copy.deepcopy(entry);e['_sampling_policy']=self.entry('narrow_recentered')['_sampling_policy']
            for c in [0,99,100,101,2000]:self.assertEqual(parameter_plan(e,c),original(entry,c))


if __name__=='__main__':unittest.main()
