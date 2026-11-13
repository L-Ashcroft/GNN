import copy
import unittest
from src.supervision_fusion import SupervisionRouter


def fixture(source='A', topology='A__AMP', name='VoltageGain'):
    spec={'source_dataset': source, 'metric_mapping': [
        {'slot':0,'semantic_id':topology+'::'+name,'unit':'dB','original_name':name}]}
    row={'source_dataset':source,'topology_id':topology,'compatibility_group':topology,
         'testbench_id':topology,'PDK':'test','temperature':27.,'supply_voltage':1.8,
         'performance':[40.]+[None]*15,'mask':[True]+[False]*15,
         'semantic_ids':[topology+'::'+name]+[None]*15}
    return spec,row


class SupervisionTests(unittest.TestCase):
    def test_same_name_unit_and_slot_do_not_allow_source_mixing(self):
        a,x=fixture();b,y=fixture('B','B__AMP')
        router=SupervisionRouter({'A__AMP':a,'B__AMP':b})
        with self.assertRaisesRegex(ValueError,'Incompatible'):router.batch([x,y])
        self.assertEqual(len(list(router.batches([x,y],8))),2)

    def test_missingness_does_not_change_head(self):
        a,x=fixture();y=copy.deepcopy(x)
        y.update(performance=[None]*16,mask=[False]*16,semantic_ids=[None]*16)
        router=SupervisionRouter({'A__AMP':a})
        batch=router.batch([x,y]);self.assertEqual(len(batch['samples']),2)
        self.assertEqual(batch['target_semantic_ids'][0],'A__AMP::VoltageGain')

    def test_wrong_semantics_cannot_hide_under_true_mask(self):
        a,x=fixture();x['semantic_ids'][0]='some_other_gain'
        with self.assertRaises(ValueError):SupervisionRouter({'A__AMP':a}).batch([x])

    def test_different_conditions_need_separate_contracts(self):
        a,x=fixture();y=copy.deepcopy(x);y['supply_voltage']=1.2
        with self.assertRaisesRegex(ValueError,'Incompatible'):SupervisionRouter({'A__AMP':a}).batch([x,y])


if __name__=='__main__':unittest.main()
