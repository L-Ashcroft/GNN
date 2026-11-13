"""Fault simulations; no actual trainer is killed, spawned or checkpoint-loaded."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment1_recovery import recovery_gate,retryable_status,RecoveryController
from validate_experiment1_resume import validate_cursor,recipe_digest

POLICY={'minimum_available_RAM_GiB':3,'minimum_disk_GiB':10,'resource_stable_seconds':120,
        'death_grace_seconds':60,'cooldown_seconds':900,'max_restarts_24h':3}

class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.context={'status':{'state':'training','test_used':False},'training_alive':False,'other_training':False,
            'test_seen':False,'stop_requested':False,'fingerprints_ok':True,'attempts_24h':0,
            'seconds_since_attempt':9999,'available_ram':4*2**30,'free_disk':30*2**30,
            'resource_stable_seconds':180,'dead_seconds':120,'checkpoint_exists':True,'checkpoint_age':60}
    def gate(self,**changes):return recovery_gate(dict(self.context,**changes),POLICY)
    def test_stable_recoverable_exit_is_eligible(self):self.assertEqual(self.gate(),'eligible_for_checkpoint_validation')
    def test_alive_training_never_restarted(self):
        self.assertEqual(self.gate(training_alive=True),'training_alive')
        controller=RecoveryController.__new__(RecoveryController)
        controller.state={'test_seen':False,'dead_since':None,'stable_since':None}
        controller.alive=Mock(return_value=True);controller.save=Mock(return_value={'state':'training_alive'})
        controller.training_processes=Mock(side_effect=AssertionError('Must not search/spawn while expected worker alive'))
        self.assertEqual(controller.step(self.context['status'])['state'],'training_alive')
    def test_second_training_blocks_duplicate_launch(self):self.assertEqual(self.gate(other_training=True),'other_training_present')
    def test_source_and_stop_guards(self):
        self.assertEqual(self.gate(fingerprints_ok=False),'fingerprint_mismatch')
        self.assertEqual(self.gate(stop_requested=True),'user_stop_requested')
    def test_resources_must_stay_healthy(self):
        self.assertEqual(self.gate(available_ram=2*2**30),'waiting_RAM')
        self.assertEqual(self.gate(free_disk=2*2**30),'waiting_disk')
        self.assertEqual(self.gate(resource_stable_seconds=30),'waiting_stable_resources')
        self.assertEqual(self.gate(dead_seconds=0),'waiting_exit_confirmation')
    def test_checkpoint_missing_or_still_writing(self):
        self.assertEqual(self.gate(checkpoint_exists=False),'checkpoint_missing')
        self.assertEqual(self.gate(checkpoint_age=1),'waiting_stable_checkpoint')
    def test_bounded_retry(self):
        self.assertEqual(self.gate(attempts_24h=3),'restart_limit_reached')
        self.assertEqual(self.gate(seconds_since_attempt=60),'waiting_cooldown')
    def test_testset_and_completion_never_replayed(self):
        self.assertEqual(self.gate(test_seen=True),'test_phase_requires_manual_review')
        for state in ['complete','testing_once']:
            self.assertFalse(retryable_status({'state':state})[0])
    def test_numeric_and_integrity_errors_not_auto_retried(self):
        for message in ['Nonfinite observed prediction','recipe mismatch','Invalid validation score','CUDA illegal memory access','unclassified crash']:
            self.assertFalse(retryable_status({'state':'error','error':message,'error_type':'RuntimeError'})[0])
    def test_transient_memory_failure_and_pause_are_retryable(self):
        self.assertTrue(retryable_status({'state':'error','error':'CUDA out of memory','error_type':'OutOfMemoryError'})[0])
        self.assertTrue(retryable_status({'state':'paused','reason':'available RAM below 256 MiB'})[0])
    def test_creation_time_detects_pid_reuse(self):
        controller=RecoveryController.__new__(RecoveryController);controller.state={'expected_pid':7,'expected_create_time':1.}
        fake=Mock();fake.create_time.return_value=2.
        with patch('experiment1_recovery.psutil.Process',return_value=fake):self.assertFalse(controller.alive())
    def checkpoint(self):
        recipe={'config':{'epochs':100,'batch_size':256}}
        checkpoint={key:{} for key in ['model','optimizer','scheduler','coverage_slots']}
        checkpoint.update(recipe_digest=recipe_digest(recipe),coverage_domains=[],next_epoch=0,next_batch=0,partial=None,
                          history=[],best=float('-inf'),best_epoch=None,bad=0,rng_cpu=None,rng_cuda=[])
        return recipe,checkpoint
    def test_cursor_and_recipe_integrity(self):
        recipe,c=self.checkpoint();validate_cursor(c,recipe,400083)
        for field,value in [('recipe_digest','wrong'),('next_batch',2000),('next_epoch',101),('history',[{}])]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_cursor(bad,recipe,400083)
    def test_partial_counters_and_finiteness(self):
        recipe,c=self.checkpoint();c['next_batch']=2;c['partial']={'samples':512,'grad_count':2,'loss_values':{'g':[1.,2]}}
        validate_cursor(c,recipe,400083)
        c['partial']['samples']=256
        with self.assertRaises(ValueError):validate_cursor(c,recipe,400083)
        c['partial']['samples']=512;c['partial']['loss_values']['g'][0]=float('nan')
        with self.assertRaises(ValueError):validate_cursor(c,recipe,400083)

if __name__=='__main__':unittest.main(verbosity=2)
