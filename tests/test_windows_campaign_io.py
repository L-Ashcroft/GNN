import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from src.common import ROOT
from src.windows_campaign_io import atomic_json,replace_with_retry


@unittest.skipUnless(os.name=='nt','Windows sharing behavior')
class WindowsIOTests(unittest.TestCase):
    def test_real_reader_lock_is_retried(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as folder:
            path=Path(folder)/'status.json';path.write_text('{"old":1}')
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateFileW.restype=ctypes.c_void_p
            kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            # Allow reading, but hold back delete sharing, reproducing editor/reader contention.
            handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            thread=threading.Timer(.25,lambda:kernel.CloseHandle(handle));thread.start()
            try:atomic_json(path,{'new':2})
            finally:thread.join()
            self.assertEqual(json.loads(path.read_text()),{'new':2})

    def test_unrelated_io_error_not_retried(self):
        with patch('src.windows_campaign_io.os.replace',side_effect=FileNotFoundError('missing')) as replace:
            with self.assertRaises(FileNotFoundError):replace_with_retry(ROOT/'missing',ROOT/'missing2')
            self.assertEqual(replace.call_count,1)

    def test_persistent_sharing_error_preserves_source(self):
        error=PermissionError('held');error.winerror=32
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as folder:
            source=Path(folder)/'a';target=Path(folder)/'b';source.write_text('new');target.write_text('old')
            with patch('src.windows_campaign_io.os.replace',side_effect=error):
                with self.assertRaises(PermissionError):replace_with_retry(source,target,timeout=0)
            self.assertEqual(source.read_text(),'new');self.assertEqual(target.read_text(),'old')


if __name__=='__main__':unittest.main()
