import unittest
import numpy as np
from src.analoggym_generation import plan_parameters, dut_graph, first_unity, edge_metrics
from src.netlist import evaluate


class GenerationTests(unittest.TestCase):
    def test_fixed_seed_and_extendable_plan(self):
        short = plan_parameters(2,42); long = plan_parameters(3,42)
        self.assertEqual(short, long[:2])
        self.assertNotEqual(long[1],long[2])
        self.assertEqual(long[0]['m_c0'],1)

    def test_geometry_mos_ports_and_cap_case(self):
        graph = dut_graph(); params = plan_parameters(1,42)[0]
        self.assertEqual(graph['num_devices'],26)
        self.assertEqual(sum(d['model'] in {'nmos','pmos'} for d in graph['devices']),24)
        self.assertEqual(graph['num_edges'],24*6+2)
        matrix = [[None if e is None else evaluate(e,params) for e in row] for row in graph['feature_expressions']]
        self.assertAlmostEqual(matrix[-2][0],30e-6)
        self.assertEqual(matrix[-2][2],1)
        self.assertEqual(matrix[-1][2],12)
        self.assertAlmostEqual(matrix[0][0],params['mosfet_11_1_w_gmf2_pmos']*1e-6)

    def test_unity_crossing_no_fallback_zero(self):
        with self.assertRaises(ValueError): first_unity(np.array([1.,10.]),np.array([.5,.1]))
        f, phase = first_unity(np.array([1.,100.]),np.array([10.+0j,.1+0j]))
        self.assertAlmostEqual(f,10.)
        self.assertEqual(phase,0.)

    def test_transient_never_settles_is_rejected(self):
        t = np.linspace(0,1,101)
        with self.assertRaises(ValueError): edge_metrics(t,np.full(101,.3),0,1,True)


if __name__ == '__main__': unittest.main()
