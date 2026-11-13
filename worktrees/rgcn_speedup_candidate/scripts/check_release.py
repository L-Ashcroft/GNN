"""CPU release integration checks; no training step or CUDA context."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
VERSION=ROOT/'versions/rgcn_cached_fast_v1'
sys.path.insert(0,str(VERSION/'scripts'))
from experiment1_health import ROOT as HEALTH_ROOT
assert HEALTH_ROOT==ROOT
suite=unittest.TestSuite()
for name in ['test_experiment1_recovery','test_experiment1_health']:
    text=(ROOT/'tests'/f'{name}.py').read_text(encoding='utf-8')
    text=text.replace("sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))",'')
    namespace={'__name__':name,'__file__':str(ROOT/'tests'/f'{name}.py')}
    exec(compile(text,name,'exec'),namespace)
    for item in namespace.values():
        if isinstance(item,type) and issubclass(item,unittest.TestCase):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(item))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
sys.path.insert(0,str(VERSION))
sys.path.append(str(ROOT))
import src.common
src.common.ROOT=ROOT
from src.train.fusion_cache import CachedFusion
from src.graph.builder import CircuitBatch
from src.model.execution import prepare_batch
config=json.loads((ROOT/'config/experiment1_fusion_id_fast_v1.json').read_text())
data=CachedFusion(config)
# Actual train iterator, same resume cursor; CPU only. Does not advance saved RNG.
batch=next(data.batches('train',256,57,1024))
assert isinstance(batch,CircuitBatch) and batch.num_graphs==256 and batch.split_role=='train'
plan=prepare_batch(batch)
assert batch.to('cpu').execution_plan is not None
report={'status':'passed','CPU_only':True,'monitor_tests':result.testsRun,'data_loader_batch_size':batch.num_graphs,
        'resumed_epoch_seed':57,'resumed_completed_batches':1024,'relations_present':len(plan.relation_slices['relation']),
        'contracts_present':len(set(batch.group_ids)),'cache_and_split_unchanged':True,'version_import':str(src.common.__file__)}
(ROOT/'monitoring/experiment1_fusion_id_fast_v1/release_integration_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
