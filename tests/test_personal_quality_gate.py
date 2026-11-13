import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
import numpy as np
from src.common import ROOT,rel
from src.analoggym_personal_quality import apply_quality


class QualityGateTests(unittest.TestCase):
    def test_discontinuous_sample_quarantined_without_rewriting_archive_or_targets(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as folder:
            archive=Path(folder)/'sample.zip'
            f=np.logspace(-1,9,1001);gain=np.where(np.arange(1001)%2,1e-4,1e4)
            stream=io.StringIO();np.savetxt(stream,np.column_stack([f,gain,np.zeros(1001)]));wave=stream.getvalue().encode()
            with zipfile.ZipFile(archive,'w') as z:z.writestr('ac_wave.dat',wave)
            before=archive.read_bytes()
            original={'accepted':True,'failure':None,'performance':[1.,2.], 'archive':rel(archive),
                'artifacts_sha256':{'ac_wave.dat':hashlib.sha256(wave).hexdigest()}}
            result=apply_quality({'family':'Amplifier'},original)
            self.assertFalse(result['accepted']);self.assertTrue(original['accepted'])
            self.assertEqual(result['performance'],original['performance'])
            self.assertEqual(archive.read_bytes(),before)
            self.assertTrue(result['ac_quality_diagnostics']['review_required'])

    def test_existing_failure_is_never_promoted(self):
        result={'accepted':False,'failure':'convergence error'}
        self.assertIs(apply_quality({'family':'Amplifier'},result),result)

    def test_ldo_not_misinterpreted_as_amplifier_ac_gain(self):
        result=apply_quality({'family':'LDO'},{'accepted':True,'failure':None})
        self.assertTrue(result['accepted']);self.assertIsNone(result['ac_quality_diagnostics'])


if __name__=='__main__':unittest.main()
