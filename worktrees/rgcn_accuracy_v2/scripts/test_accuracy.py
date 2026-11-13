"""Guarded CPU tests; no CUDA context or concurrent GPU benchmark."""
from pathlib import Path
import sys
import os
import threading
import unittest
import json
import io
import psutil
PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1];sys.path.insert(0,str(PACKAGE))
if psutil.virtual_memory().available<1024*2**20:
    (PACKAGE/'results/cpu_tests.json').write_text(json.dumps({'status':'deferred','reason':'Need 1 GiB available for CPU tests'}))
    raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
stop=threading.Event();measurements=[]
def guard():
    process=psutil.Process()
    while not stop.wait(.1):
        available=psutil.virtual_memory().available;measurements.append((process.memory_info().rss,available))
        if available<512*2**20:
            (PACKAGE/'results/cpu_tests.json').write_text(json.dumps({'status':'aborted_for_RAM','reason':'Only CPU diagnostic exits below 512 MiB'}))
            os._exit(86)
threading.Thread(target=guard,daemon=True).start()
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(PACKAGE/'tests'))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);stop.set()
(PACKAGE/'results/cpu_tests.log').write_text(stream.getvalue(),encoding='utf-8')
report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'skipped':[str(t)+': '+reason for t,reason in result.skipped],
        'peak_RSS_MiB':max((v[0] for v in measurements),default=0)/2**20,
        'minimum_available_RAM_MiB':min((v[1] for v in measurements),default=0)/2**20,'CUDA_context_created':False}
(PACKAGE/'results/cpu_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(stream.getvalue());print(json.dumps(report,indent=2))
if not result.wasSuccessful():raise SystemExit(1)
