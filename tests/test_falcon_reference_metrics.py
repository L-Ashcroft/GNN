"""Reference-golden checks for changed evaluation semantics, not model accuracy."""
from pathlib import Path
import ast
import unittest
import numpy as np
from src.falcon_reference_metrics import relative_errors, regression, target_report, FalconReferenceAccumulator

ROOT = Path(__file__).resolve().parents[1]

class ReferenceMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'third_party/FALCON/utils/compute_metrics.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_relative_errors')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'official_compute_relative_errors', 'exec'), namespace)
        cls.official = staticmethod(namespace['compute_relative_errors'])

    def test_official_formula_signed_zero_tiny_and_missing(self):
        y = np.array([[1e-12, -20., 0., 2.], [1., 4., 0., -1.]])
        p = np.array([[2e-12, -18., 99., 4.], [1.2, 3., 0., -1.1]])
        mask = np.array([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
        before = mask.copy()
        actual = relative_errors(p, y, mask)
        expected = self.official(p, y, mask)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask, before)

    def test_no_relative_target_is_undefined_not_zero(self):
        r, valid, sample = relative_errors([[7., 8.]], [[0., 0.]], [[True, False]])
        self.assertTrue(np.isnan(sample[0]))
        self.assertEqual(valid.sum(), 0)
        self.assertEqual(r.sum(), 0.)

    def test_observed_nonfinite_raises_but_missing_is_not_used(self):
        with self.assertRaises(ValueError):
            relative_errors([[np.inf]], [[1.]], [[True]])
        _, _, sample = relative_errors([[np.nan, 2.]], [[0., 1.]], [[False, True]])
        self.assertTrue(np.isfinite(sample[0]))

    def test_sklearn_constant_target_policy_is_explicit(self):
        self.assertEqual(regression([2., 2.], [2., 2.])['r2'], 1.)
        out = regression([2., 2.], [3., 3.])
        self.assertEqual(out['r2'], 0.)
        self.assertIsNone(out['r2_mathematical'])
        self.assertIsNone(regression([2.], [2.])['r2'])

    def test_reference_zero_filter_does_not_delete_physical_observation(self):
        out = target_report([(np.array([0., 1., 2.]), np.array([10., 1., 2.]))])
        self.assertEqual(out['reference_nonzero']['mae'], 0.)
        self.assertAlmostEqual(out['all_observed']['mae'], 10. / 3.)
        self.assertEqual(out['observed_count'], 3)
        self.assertEqual(out['observed_zero_count'], 1)

    def test_sample_averaging_is_not_scalar_or_task_averaging(self):
        a = FalconReferenceAccumulator()
        rules = {'0': {'semantic_id': 'x', 'unit': 'dB', 'original_name': 'x'},
                 '1': {'semantic_id': 'y', 'unit': 'V', 'original_name': 'y'}}
        a.update([[1., 3.], [2., 0.]], [[1., 1.], [1., 0.]], [[1, 1], [1, 0]],
                 ['a', 'b'], 'source', 'top', 'contract', rules)
        s, _, _, rows = a.finish()
        expected = np.mean([(0. + 2. / (1. + 1e-8)) / 2, 1. / (1. + 1e-8)]) * 100
        self.assertAlmostEqual(s['overall']['untrimmed_sample_mre']['mean_pct'], expected)
        self.assertNotAlmostEqual(expected, (0. + 2. + 1.) / 3 * 100)
        self.assertEqual(rows[0]['relative_valid_metrics'], 2)  # includes dB by reference convention
        with self.assertRaises(ValueError):
            a.update([[1., 1.]], [[1., 1.]], [[1, 1]], ['a'], 'source', 'top', 'contract', rules)

    def test_native_percent_restoration_before_epsilon(self):
        a = FalconReferenceAccumulator()
        rules = {'0': {'semantic_id': 'de', 'unit': '1', 'native_unit': '%', 'native_multiplier': 100.,
                       'original_name': 'DrainEfficiency'}}
        a.update([[.95]], [[1.]], [[True]], ['a'], 'FALCON', 'top', 'g', rules)
        summary, rows, pooled, _ = a.finish()
        self.assertAlmostEqual(summary['overall']['untrimmed_sample_mre']['mean_pct'], 5. / (100. + 1e-8) * 100)
        self.assertEqual(rows[0]['native_unit'], '%')
        self.assertEqual(pooled[0]['reference_nonzero']['mae'], 5.)

    def test_real_frozen_falcon_batch_matches_official(self):
        import pyarrow.parquet as pq
        path = ROOT / 'artifacts/experiment1_rgcn_local_head_v1/fixed_test/FALCON__CCVCO.parquet'
        rows = next(pq.ParquetFile(path).iter_batches(batch_size=32)).to_pylist()
        p = np.asarray([r['prediction'] for r in rows], dtype=np.float64)
        y = np.asarray([r['target'] for r in rows], dtype=np.float64)
        mask = np.asarray([r['mask'] for r in rows], dtype=bool)
        actual, expected = relative_errors(p, y, mask), self.official(p, y, mask)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

if __name__ == '__main__':
    unittest.main()
