import io
import pickle
import unittest
import numpy as np
from scripts.probe_analoggym_replay import RestrictedReplayUnpickler
from scripts.audit_analoggym_replay import action_schema, BASE
from src.load_analoggym import iter_replay_candidates


class AuditTests(unittest.TestCase):
    def test_restricted_unpickler_rejects_executable_global(self):
        with self.assertRaisesRegex(pickle.UnpicklingError, 'Forbidden global'):
            RestrictedReplayUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_numpy_array_roundtrip_without_upstream_import(self):
        expected = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        actual = RestrictedReplayUnpickler(io.BytesIO(pickle.dumps(expected, protocol=4))).load()
        np.testing.assert_array_equal(actual, expected)

    def test_source_parameter_order_and_quantization_schema(self):
        names, ints, mapping, bounds = action_schema(BASE / 'LDO_TB.py', 'GraphLDOtestbench')
        self.assertEqual(len(names), 21)
        self.assertEqual(names[9:12], ['W_power', 'L_power', 'M_power'])
        self.assertEqual(names[-1], 'M_CL')
        self.assertIn('M_CL', ints)
        self.assertEqual(mapping['Ib'], 'current_0_bias')
        self.assertEqual(bounds['action_space_high'][-1], 100)

    def test_default_replay_loader_blocks_quarantined_rows(self):
        with self.assertRaises(ValueError):
            next(iter_replay_candidates())


if __name__ == '__main__': unittest.main()
