"""Run only CPU process-identity/resource-gate checks, without importing torch."""
import io
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_gpu_gate.py')
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
output=ROOT/'results/gpu_acceptance';output.mkdir(exist_ok=True)
(output/'gate_tests.log').write_text(stream.getvalue(),encoding='utf-8')
(output/'gate_tests.json').write_text(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),
    'errors':len(result.errors),'passed':result.wasSuccessful(),'torch_imported':'torch' in sys.modules,
    'CUDA_work_started':False},indent=2),encoding='utf-8')
print(stream.getvalue())
if not result.wasSuccessful():raise SystemExit(1)
