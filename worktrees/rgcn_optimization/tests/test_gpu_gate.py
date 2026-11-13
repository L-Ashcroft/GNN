"""CPU-only guard tests; no torch import or CUDA context."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import gpu_preflight as gate

class FakeProcess:
    def is_running(self):return True
    def create_time(self):return 100.125
    def name(self):return 'python.exe'
    def cmdline(self):return ['python.exe','scripts/train_experiment1.py']
    def cwd(self):return r'D:\EDA'

class GateTests(unittest.TestCase):
    def check_entrypoint_defers(self,filename,arguments):
        path=Path(__file__).resolve().parents[1]/'scripts'/filename
        spec=importlib.util.spec_from_file_location('_guard_'+path.stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch_before='torch' in sys.modules
        with patch.object(gate,'check_environment',return_value={'eligible_for_CUDA':False}), \
             patch.object(module,'write') as writer,patch.object(module,'read',return_value={'pid':27996}), \
             patch.object(sys,'argv',[str(path)]+arguments):
            with self.assertRaises(SystemExit) as result:module.main()
        self.assertEqual(result.exception.code,75)
        self.assertEqual(writer.call_args.args[1]['status'],'deferred')
        self.assertEqual('torch' in sys.modules,torch_before)

    def test_cuda_validation_entry_does_not_treat_deferred_as_pass(self):
        self.check_entrypoint_defers('validate.py',['--stage','gate_unit_test','--device','cuda','--batch-size','256'])

    def test_cuda_benchmark_entry_does_not_treat_deferred_as_pass(self):
        self.check_entrypoint_defers('benchmark.py',['--variant','reference','--device','cuda','--batch-size','256'])

    def test_original_identity_requires_creation_time(self):
        with patch.object(gate.psutil,'Process',return_value=FakeProcess()):
            same=gate.process_identity(7,100.125);reused=gate.process_identity(7,99.)
        self.assertTrue(same['same_creation_time']);self.assertFalse(reused['same_creation_time'])
        self.assertTrue(gate.blocking_reasons(same,[],8*2**30,4))
        self.assertEqual(gate.blocking_reasons(reused,[],8*2**30,4),[])

    def test_unknown_identity_is_blocked(self):
        self.assertIn('original_process_identity_unverified',gate.blocking_reasons({'identity_unverified':True},[],8*2**30,4))

    def test_restarted_training_is_blocked(self):
        self.assertIn('workspace_training_command_still_running',gate.blocking_reasons({'exists':False},[{'pid':8}],8*2**30,4))

    def test_memory_threshold_and_original_alive(self):
        gone={'exists':False};live={'exists':True,'same_creation_time':True}
        self.assertTrue(gate.blocking_reasons(gone,[],4*2**30-1,4))
        self.assertEqual(gate.blocking_reasons(gone,[],4*2**30,4),[])
        self.assertTrue(gate.blocking_reasons(live,[],8*2**30,4))
        self.assertTrue(gate.blocking_reasons(gone,[],5*2**30,6))

    def test_command_token_not_embedded_text(self):
        self.assertTrue(gate.is_training_command(['python.exe','-u',r'scripts\train_experiment1.py']))
        self.assertFalse(gate.is_training_command(['python.exe','-c',"print('train_experiment1.py')"]))
        self.assertFalse(gate.is_training_command(['python.exe','scripts/gpu_preflight.py']))

if __name__=='__main__':unittest.main(verbosity=2)
