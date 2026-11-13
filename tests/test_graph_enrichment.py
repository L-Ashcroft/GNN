import math
import unittest
from src.netlist import parse_spectre,make_graph,feature_expressions,FEATURE_NAMES
from src.graph_enrichment import enrich_graph,vocabularies,transform_features


def graph(text):
    g=make_graph(parse_spectre(text));g.update(feature_names=FEATURE_NAMES,feature_expressions=feature_expressions(g))
    return g


class GraphFeatureTests(unittest.TestCase):
    def test_ground_source_preserves_distinct_nodes_and_device_degree(self):
        g=graph('M0 (out in GND GND) nmos w=2u l=1u\nV0 (GND 0) vsource dc=0 type=dc\nV1 (VCC 0) vsource dc=1.8 type=dc')
        s={'source_dataset':'FALCON','defaults':{}}
        e=enrich_graph(g,s,vocabularies([g]));self.assertEqual(e['nets'],g['nets'])
        a={x['net']:x for x in e['node_annotations']}
        self.assertIn('ground',a['GND']['roles']);self.assertTrue(a['GND']['has_verified_evidence'])
        self.assertIn('positive_supply',a['VCC']['roles'])
        self.assertEqual(e['node_features'][e['nets'].index('GND')][0],2)
        self.assertEqual(e['node_features'][e['nets'].index('GND')][e['node_feature_names'].index('terminal_count:nmos:S')],1)
        self.assertEqual(e['num_message_edges'],2*g['num_edges'])
        self.assertEqual(e,enrich_graph(e,s,vocabularies([g])))

    def test_reverse_relations_keep_tied_terminal_identity(self):
        g=graph('M0 (out in 0 0) nmos w=2u l=1u')
        e=enrich_graph(g,{'source_dataset':'AICircuit','defaults':{}},vocabularies([g]));m=e['message_passing']
        i=next(i for i,p in enumerate(m['edge_port_pairs']) if p==['S','B']);j=m['reverse_edge_index'][i]
        self.assertEqual(m['edge_port_pairs'][j],['B','S'])
        self.assertNotEqual(m['edge_type'][i],m['edge_type'][j])
        self.assertEqual(m['edge_index'][0][i],m['edge_index'][1][i])

    def test_ac_excitation_at_zero_dc_is_not_ground(self):
        g=graph('V0 (Vin 0) vsource dc=0 mag=1 type=dc\nR0 (Vin out) resistor r=1k')
        e=enrich_graph(g,{'source_dataset':'FALCON','defaults':{}},vocabularies([g]))
        a={x['net']:x for x in e['node_annotations']}
        self.assertNotIn('ground',a['Vin']['roles']);self.assertIn('input',a['Vin']['roles'])

    def test_zero_cap_and_missing_nf_are_explicit(self):
        values,known,zero=transform_features([[0.,None,4.,2e-6]],['capacitance_F','finger_count','multiplicity','width_m'])
        self.assertEqual(values[0][:3],[0.,None,2.]);self.assertEqual(known[0],[True,False,True,True])
        self.assertEqual(zero[0],[True,False,False,False]);self.assertAlmostEqual(values[0][3],math.log10(2e-6))

    def test_unnamed_drain_feed_is_supply_not_gate_bias(self):
        g=graph('V0 (rail 0) vsource dc=1.2 type=dc\nL0 (rail out) inductor l=1n\nM0 (out gate 0 0) nmos w=2u l=1u\nV1 (gate 0) vsource dc=0.6 type=dc')
        e=enrich_graph(g,{'source_dataset':'FALCON','defaults':{}},vocabularies([g]))
        a={x['net']:x for x in e['node_annotations']}
        self.assertIn('positive_supply',a['rail']['roles'])
        self.assertNotIn('positive_supply',a['gate']['roles'])


if __name__=='__main__':unittest.main()
