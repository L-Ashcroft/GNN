import re
import unittest
from pathlib import Path
import pandas as pd
from src.common import read
from src.analoggym_campaign import parameter_plan,evaluate
from src.analoggym_personal_protocols import nested_order


class PersonalCampaignTests(unittest.TestCase):
    def test_nested_budget_order_is_deterministic_and_topology_balanced(self):
        frame=pd.DataFrame([{'sample_id':f'{topology}:{i}','topology_id':topology} for topology in ['a','b','c'] for i in range(5)])
        a=nested_order(frame);b=nested_order(frame.sample(frac=1.,random_state=8))
        self.assertEqual(a,b);self.assertEqual(len(set(a)),15)
        self.assertEqual({x.split(':')[0] for x in a[:3]},{'a','b','c'})
        self.assertTrue(set(a[:5])<=set(a[:10]))

    def test_actual_sampler_retains_aliases_and_physical_geometry_limits(self):
        path=Path('data/generated/analoggym_personal_20k_v1/recipe.json')
        if not path.exists():self.skipTest('Personal campaign not prepared')
        recipe=read(path);self.assertEqual(len(recipe['entries']),10)
        for entry in recipe['entries']:
            for candidate in [0,1,2,7,10,19,42]:
                params=parameter_plan(entry,candidate)
                self.assertEqual(params,parameter_plan(entry,candidate))
                for key,value in params.items():
                    if re.search(r'(^w_|_w_)',key):self.assertTrue(.42<=value<=100.)
                    if re.search(r'(^l_|_l_)',key):self.assertTrue(.15<=value<=30.)
                    if re.search(r'(^m_|_m_)',key):self.assertEqual(value,round(value))
                for key,expression in entry['definitions'].items():
                    try:evaluate(expression,{})
                    except KeyError:self.assertEqual(params[key],evaluate(expression,params))


if __name__=='__main__':unittest.main()
