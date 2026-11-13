import copy
from pathlib import Path
import unittest
import torch
from src.common import ROOT,read
from src.personal_gnn_data import PersonalDataset,context_values
from src.model.personal_network import PersonalEdgeRGCN,initialize_backbone
from src.model.network import EdgeCentricRGCN
from src.train.stage import build_optimizer
from src.train.loop import fit


class PersonalGNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config=read(ROOT/'config/personal_rgcn.json')
        if not (ROOT/config['derived']/'manifest.json').exists():raise unittest.SkipTest('Prepare personal model data first')
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.data=PersonalDataset(config);cls.rows=list(cls.data.rows('train',limit_per_topology=1))

    def model(self):
        return PersonalEdgeRGCN(self.data.schema,self.data.tasks,hidden_dim=16,num_layers=2,dropout=0.,
            use_domain_emb=True,use_film=True).eval()

    def test_all_graphs_roles_ports_and_substrate(self):
        for row in self.rows:
            graph,_=self.data.materialize(row)
            original=read(ROOT/self.data.catalog[row['topology_id']]['source_graph'])
            for a,b in zip(graph['devices'],original['devices']):
                self.assertEqual(a['nets'],b['nets']);self.assertEqual(a['ports'],b['ports'])
            roles={a['net']:a['roles'] for a in graph['node_annotations']}
            self.assertTrue(any('ground' in r for r in roles.values()))
            self.assertTrue(any('positive_supply' in r for r in roles.values()))
            if row['circuit_family']=='LDO':
                self.assertIn('ground',roles['vss'])
                resistor=next(d for d in graph['devices'] if d['name']=='xrfb')
                self.assertEqual(resistor['ports'],['P','N','B'])
                self.assertEqual(resistor['model'],'resistor')
                self.assertIn('sky130',resistor['source_device_model'])
        self.assertEqual(len(self.data.schema['node_features']),35)
        self.assertEqual(len(self.data.schema['relations']),52)

    def test_labels_not_used_as_input_or_prediction(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:row['performance']=[None if v is None else v*1.1 for v in row['performance']]
        a=self.data.tensorize(self.rows,'train');b=self.data.tensorize(rows,'train')
        for key in ['x','edge_attr','conditions']:torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)
        model=self.model()
        with torch.no_grad():torch.testing.assert_close(model(a),model(b),rtol=0,atol=0)
        for row in rows:
            row['performance']=[None]*16;row['mask']=[False]*16;row['semantic_ids']=[None]*16
        unlabelled=self.data.tensorize(rows)
        with torch.no_grad():torch.testing.assert_close(model(a),model(unlabelled),rtol=0,atol=0)

    def test_bias_reaches_network(self):
        row=next(r for r in self.rows if r['topology_id']=='ldo_simple');changed=copy.deepcopy(row)
        index=changed['parameter_names'].index('vb');changed['parameters'][index]*=1.2
        a=self.data.tensorize([row],'train');b=self.data.tensorize([changed],'train')
        self.assertNotEqual(context_values(row),context_values(changed))
        torch.testing.assert_close(a.edge_attr,b.edge_attr,rtol=0,atol=0)
        model=self.model()
        with torch.no_grad():self.assertGreater(float((model(a)-model(b)).abs().max()),0.)

    def test_roles_and_semantics_guard(self):
        with self.assertRaises(ValueError):self.data.tensorize(self.rows,'test')
        changed=copy.deepcopy(self.rows[0]);changed['semantic_ids'][0]='unapproved_metric'
        with self.assertRaises(ValueError):self.data.tensorize([changed])

    def test_batch_to_and_node_permutation(self):
        batch=self.data.tensorize(self.rows,'train').to('cpu');model=self.model()
        changed=copy.copy(batch);perm=torch.randperm(len(batch.x));inverse=torch.argsort(perm)
        changed.x=batch.x[perm];changed.graph_index=batch.graph_index[perm];changed.edge_index=inverse[batch.edge_index]
        with torch.no_grad():torch.testing.assert_close(model(batch),model(changed),atol=2e-5,rtol=2e-5)

    def test_pretrained_name_alignment_and_new_relations(self):
        model=self.model();old_schema=copy.deepcopy(self.data.schema)
        old_schema['node_features']=[n for n in old_schema['node_features'] if n!='terminal_count:resistor:B']
        old_schema['relations']=[n for n in old_schema['relations'] if not (n.startswith('resistor:') and 'B' in n)]
        original=EdgeCentricRGCN(old_schema,self.data.tasks,hidden_dim=16,num_layers=2,dropout=0.)
        report=initialize_backbone(model,{'schema':old_schema,'model':original.state_dict()})
        self.assertEqual(len(report['added_relations']),4)
        self.assertFalse(report['target_readouts_transferred'])
        for i,name in enumerate(old_schema['relations']):
            j=model.schema['relations'].index(name)
            torch.testing.assert_close(model.layers[0].relation_weight[j],original.layers[0].relation_weight[i])
        build_optimizer(model,'finetune_head')
        self.assertFalse(model.node_encoder.weight.requires_grad)
        self.assertTrue(model.condition_encoder[0].weight.requires_grad)

    def test_real_loader_and_fit_pipeline(self):
        model=self.model();optimizer=build_optimizer(model)
        result=fit(model,
            lambda epoch:self.data.batches('train',16,seed=42+epoch,limit_per_topology=2),
            lambda:self.data.batches('validation',16,limit_per_topology=1),
            optimizer,epochs=1,patience=1)
        self.assertEqual(result['best_epoch'],0)
        self.assertEqual(result['history'][0]['training']['samples'],20)
        self.assertEqual(result['history'][0]['validation']['samples'],10)


if __name__=='__main__':unittest.main()
