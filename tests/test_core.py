import unittest
from src.netlist import parse_spectre, make_graph, evaluate, feature_expressions
from src.merge_sample import balanced_budgets
from src.convert import find_ambiguous_input_rows
from src.common import ROOT
from src.load_falcon import discover
import pandas as pd


class CoreTests(unittest.TestCase):
    def test_conflicting_input_groups_are_quarantined_not_averaged(self):
        frame = pd.DataFrame({'x': [1., 1., 2., 2.], 'gain': [10., 11., 8., 8.]})
        self.assertEqual(find_ambiguous_input_rows(frame, ['x'], ['gain']), {1, 2})

    @unittest.skipUnless((ROOT / 'third_party/FALCON/dataset/LNA/CLNA/values.yaml').exists(), 'Official snapshot not acquired')
    def test_official_fixed_parameters_are_consumed(self):
        spec = next(s for s in discover() if s['source_topology'] == 'CLNA')
        self.assertAlmostEqual(spec['defaults']['W'], 5e-6)
        self.assertEqual(spec['defaults']['VDD'], 1.2)
        self.assertEqual(spec['defaults']['fin'], 30e9)
        self.assertTrue(spec['default_files'])

    def test_mos_tied_bulk_keeps_four_roles(self):
        g = make_graph(parse_spectre('M0 (out in 0 0) nmos w=W l=45n'))
        self.assertEqual(g['num_edges'], 6)
        self.assertEqual(g['num_nodes'], 3)
        self.assertEqual([e['port'] for e in g['device_terminals']], ['D', 'G', 'S', 'B'])
        self.assertEqual(g['device_terminals'][2]['net'], g['device_terminals'][3]['net'])
        self.assertEqual([e['ports'] for e in g['edges']], [['D','G'], ['D','S'], ['D','B'], ['G','S'], ['G','B'], ['S','B']])
        self.assertEqual(g['edges'][-1]['source'], g['edges'][-1]['target'])
        self.assertTrue(all(e['device'] == 0 for e in g['edges']))

    def test_parallel_devices_and_shortened_terminals_keep_identity(self):
        g = make_graph(parse_spectre('R0 (a b) resistor r=1k\nR1 (a b) resistor r=2k\nC0 (a a) capacitor c=1p'))
        self.assertEqual(g['num_nodes'], 2)
        self.assertEqual(g['num_edges'], 3)
        self.assertEqual([e['name'] for e in g['edges']], ['R0', 'R1', 'C0'])
        self.assertEqual(g['edges'][0]['source'], g['edges'][1]['source'])
        self.assertEqual(g['edges'][-1]['source'], g['edges'][-1]['target'])

    def test_balun_star_keeps_original_port_order(self):
        g = make_graph(parse_spectre('B0 (a b c) balun ratio=1'))
        self.assertEqual([e['ports'] for e in g['edges']], [['P1', 'P2'], ['P1', 'P3']])
        self.assertEqual(g['num_nodes'], 3)

    def test_hierarchy_local_nodes_are_not_shortened(self):
        text = 'subckt cell a b\nM0 (a b x 0) nmos w=2u l=45n\nends cell\nI0 (v1 v2) cell\nI1 (v3 v4) cell'
        g = make_graph(parse_spectre(text))
        self.assertEqual(g['num_devices'], 2)
        self.assertIn('I0:x', g['nets']); self.assertIn('I1:x', g['nets'])
        self.assertEqual(g['num_nets'], 7)

    def test_ground_source_is_retained(self):
        g = make_graph(parse_spectre('V0 (GND 0) vsource dc=0'))
        self.assertEqual(g['num_nets'], 2)

    def test_mutual_inductor_links_devices(self):
        g = make_graph(parse_spectre('L0 (a b) inductor l=1n\nL1 (c d) inductor l=2n\nK0 mutual_inductor coupling=0.9 ind1=L0 ind2=L1'))
        self.assertEqual(g['num_devices'], 3)
        self.assertEqual(g['num_nets'], 4)
        self.assertEqual(g['num_edges'], 2)
        self.assertEqual(g['num_device_references'], 2)
        self.assertEqual({e['target'] for e in g['device_references']}, {0, 1})

    def test_units_and_safe_arithmetic(self):
        self.assertAlmostEqual(evaluate('sqrt(Lp*Ls)*k', {'Lp': 4e-9, 'Ls': 1e-9, 'k': .8}), 1.6e-9)
        self.assertAlmostEqual(evaluate('2.5K + 3m', {}), 2500.003)
        with self.assertRaises((ValueError, KeyError)):
            evaluate('__import__("os").system("echo BAD")', {})
        with self.assertRaises(KeyError):
            evaluate('VDD', {})

    def test_strict_parse_and_budget(self):
        with self.assertRaises(ValueError):
            parse_spectre('M0 (d g s) nmos w=1u')
        with self.assertRaises(ValueError):
            parse_spectre('something unsupported')
        self.assertEqual(balanced_budgets({'a': 2, 'b': 100, 'c': 100}, 12), {'a': 2, 'b': 5, 'c': 5})


if __name__ == '__main__':
    unittest.main()
