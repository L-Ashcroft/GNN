import unittest
import numpy as np
from scripts.evaluate_analoggym_personal import ac_diagnostics


class ACSanityTests(unittest.TestCase):
    def test_smooth_amplifier_response_not_flagged(self):
        frequency=np.logspace(-1,9,1001)
        response=1e4/(1+1j*frequency/100)
        diagnostic=ac_diagnostics(np.column_stack([frequency,response.real,response.imag]))
        self.assertFalse(diagnostic['review_required'])
        self.assertEqual(diagnostic['descending_unity_crossings'],1)

    def test_repeated_gain_discontinuities_are_flagged(self):
        frequency=np.logspace(-1,9,1001);gain=np.where(np.arange(1001)%2,1e-4,1e4)
        diagnostic=ac_diagnostics(np.column_stack([frequency,gain,np.zeros(1001)]))
        self.assertTrue(diagnostic['review_required'])
        self.assertGreater(diagnostic['gain_jumps_over_20dB_per_grid_step'],100)

    def test_single_resonance_is_not_blanket_rejected(self):
        frequency=np.logspace(-1,9,1001)
        gain=np.ones(1001)*2;gain[450:500]=.5;gain[800:]=.2
        diagnostic=ac_diagnostics(np.column_stack([frequency,gain,np.zeros(1001)]))
        self.assertEqual(diagnostic['descending_unity_crossings'],2)
        self.assertFalse(diagnostic['review_required'])


if __name__=='__main__':unittest.main()
