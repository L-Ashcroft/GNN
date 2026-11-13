"""CPU-only observer checks; no training process or power settings are changed."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment1_health import runtime_findings,nonfinite_paths,history_summary,quality_findings,observer_json

class HealthChecks(unittest.TestCase):
    def setUp(self):
        self.clock=datetime(2026,10,31,tzinfo=timezone.utc)
        self.status={'state':'training','pid':7,'updated_at':self.clock.isoformat(),'training_loss':.8}
    def issues(self,status=None,identity=True,ram=2*2**30,disk=100*2**30):
        return runtime_findings(status or self.status,identity,ram,disk,self.clock)
    def test_healthy_training(self):self.assertEqual(self.issues(),[])
    def test_stale_training_but_validation_has_more_time(self):
        self.status['updated_at']=(self.clock-timedelta(seconds=400)).isoformat()
        self.assertEqual(self.issues()[0]['code'],'status_stale')
        self.status['state']='validating';self.assertEqual(self.issues(),[])
    def test_missing_identity_and_low_resources(self):
        codes={i['code'] for i in self.issues(identity=False,ram=400*2**20,disk=500*2**20)}
        self.assertEqual(codes,{'original_process_missing_or_changed','low_RAM','low_disk'})
    def test_nonfinite_is_not_silently_sanitized(self):
        self.status['training_loss']=float('nan')
        self.assertTrue(nonfinite_paths(self.status));self.assertEqual(self.issues()[0]['code'],'nonfinite_live_status')
        encoded=observer_json(self.status)
        self.assertEqual(encoded['training_loss'],{'__nonfinite__':'nan'})
        self.assertTrue(nonfinite_paths(self.status))
    def test_completed_process_missing_is_not_a_failure(self):
        self.status['state']='complete';self.assertEqual(self.issues(identity=False),[])
    def test_negative_macro_does_not_mean_every_target_failed(self):
        targets=[{'semantic_id':'FALCON__X::a','supervision_group':'g','r2':.5,'mae':1.,'count':10,'hit5':.5,'hit10':.7},
                 {'semantic_id':'FALCON__X::b','supervision_group':'g','r2':-1000.,'mae':2.,'count':10,'hit5':.1,'hit10':.2}]
        epoch={'epoch':1,'training_loss':1.,'training_samples':16,'gradient_norm_mean':2.,'gradient_norm_max':4.,
               'training_seconds':20.,'validation_seconds':4.,'validation':{'loss':1.,'macro_hit5':.3,'macro_hit10':.45,
                'macro_r2':-499.75,'samples':20,'per_target':targets}}
        result=history_summary([epoch])[0]
        self.assertEqual(result['positive_r2'],1);self.assertEqual(result['r2_below_minus100'],1)
        self.assertEqual(quality_findings([result])[0]['code'],'extreme_negative_r2')

if __name__=='__main__':unittest.main(verbosity=2)
