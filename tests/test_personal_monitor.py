import unittest
from scripts.monitor_analoggym_personal import result_errors,window_summary


class MonitorTests(unittest.TestCase):
    def test_nonfinite_or_incomplete_accepted_labels_fail(self):
        row={'accepted':True,'performance':[1.,float('nan')],'parameters_native':{'w':1.},'failure':None}
        self.assertIn('non-finite performance',result_errors(row,2))
        self.assertIn('performance dimensions mismatch',result_errors(row,3))
        row['performance']=[1.,2.];self.assertEqual(result_errors(row,2),[])

    def test_quarantined_missing_labels_are_expected(self):
        self.assertEqual(result_errors({'accepted':False,'performance':None,'parameters_native':{'w':1.}},13),[])

    def test_new_policy_yield_does_not_mix_old_failures(self):
        rows=[{'accepted':True,'sampling_policy_version':'v2'}]*3+[
            {'accepted':False,'failure':'old failure'}]*17
        summary=window_summary(rows)
        self.assertEqual(summary['attempts'],3);self.assertEqual(summary['accepted'],3)
        self.assertEqual(summary['consecutive_failures'],0)

    def test_failure_streak_stops_at_first_success(self):
        rows=[{'accepted':False,'failure':'bad'}]*10+[{'accepted':True}]
        self.assertEqual(window_summary(rows)['consecutive_failures'],10)


if __name__=='__main__':unittest.main()
