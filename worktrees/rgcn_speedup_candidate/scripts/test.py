"""Run edge-case checks with the same host-memory guard as equivalence tests."""
import io
import threading
import unittest
from validate import ROOT,guard_memory,write

stop=threading.Event();threading.Thread(target=guard_memory,args=(stop,),daemon=True).start()
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(ROOT/'results/edge_case_tests.log').write_text(stream.getvalue(),encoding='utf-8')
write(ROOT/'results/edge_case_tests.json',{'tests_run':result.testsRun,'failures':len(result.failures),
    'errors':len(result.errors),'skipped':len(result.skipped),'success':result.wasSuccessful()})
stop.set();print(stream.getvalue())
if not result.wasSuccessful():raise SystemExit(1)
