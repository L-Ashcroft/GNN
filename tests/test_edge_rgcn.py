import copy
import importlib.util
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from src.gnn.data import CircuitBatch
    from src.gnn.losses import masked_mse,supervision_loss,symlog,inverse_symlog
    from src.gnn.model import EdgeCentricRGCN,EdgeRGCNLayer


@unittest.skipUnless(HAS_TORCH,'Install requirements-gnn.txt to test model')
class EdgeRGCNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        self.schema={'node_features':['a','b'],'device_features':['w'],'relations':['D>G','G>D']}
        self.tasks={'task':{'slots':{'0':{'family':'gain','semantic_id':'task::gain'}}}}
        self.model=EdgeCentricRGCN(self.schema,self.tasks,hidden_dim=8,num_layers=2,dropout=0.)
        self.batch=CircuitBatch(torch.tensor([[1.,0.],[0.,1.],[.5,1.]]),
            torch.tensor([[0,1,1,2],[1,0,2,1]]),torch.randn(4,3),torch.tensor([0,1,0,1]),
            torch.tensor([0,0,0]),torch.tensor([[2.]+[0.]*15]),torch.tensor([[True]+[False]*15]),
            torch.ones(1,16),['task'],[['task::gain']+[None]*15],['example'])

    def test_all_edge_update_layers_receive_gradients(self):
        out=self.model(self.batch,True);loss,_=supervision_loss(out['prediction'],self.batch);loss.backward()
        for layer in self.model.layers:
            self.assertIsNotNone(layer.edge_update[0].weight.grad)
            self.assertGreater(layer.edge_update[0].weight.grad.abs().sum().item(),0.)
        self.assertFalse(torch.allclose(out['initial_edge_states'],out['edge_states']))

    def test_fixed_edge_state_ablation(self):
        model=EdgeCentricRGCN(self.schema,self.tasks,hidden_dim=8,num_layers=2,dropout=0.,update_edges=False)
        out=model(self.batch,True)
        self.assertTrue(torch.equal(out['initial_edge_states'],out['edge_states']))

    def test_node_relabeling_invariance(self):
        self.model.eval();a=self.model(self.batch)
        order=torch.tensor([2,0,1]);inverse=torch.argsort(order);b=copy.copy(self.batch)
        b.x=b.x[order];b.graph_index=b.graph_index[order];b.edge_index=inverse[b.edge_index]
        torch.testing.assert_close(a,self.model(b),rtol=1e-5,atol=1e-6)

    def test_edge_order_invariance(self):
        self.model.eval();a=self.model(self.batch);b=copy.copy(self.batch);p=torch.tensor([3,1,0,2])
        b.edge_index=b.edge_index[:,p];b.edge_attr=b.edge_attr[p];b.edge_type=b.edge_type[p]
        torch.testing.assert_close(a,self.model(b),rtol=1e-5,atol=1e-6)

    def test_attention_sums_per_graph_and_batch_isolation(self):
        self.model.eval();a=self.model(self.batch,True)
        self.assertAlmostEqual(a['node_attention'].sum().item(),1.,places=6)
        self.assertAlmostEqual(a['edge_attention'].sum().item(),1.,places=6)
        b=copy.copy(self.batch);b.x=torch.cat([b.x,b.x+2.]);b.edge_index=torch.cat([b.edge_index,b.edge_index+3],1)
        b.edge_attr=torch.cat([b.edge_attr,b.edge_attr]);b.edge_type=torch.cat([b.edge_type,b.edge_type])
        b.graph_index=torch.tensor([0,0,0,1,1,1]);b.group_ids=['task','task']
        torch.testing.assert_close(a['prediction'][0],self.model(b)[0],rtol=1e-5,atol=1e-6)

    def test_relation_specific_destination_normalization(self):
        layer=EdgeRGCNLayer(1,2,dropout=0.,update_edges=False);layer.node_norm=torch.nn.Identity()
        with torch.no_grad():
            layer.root.weight.zero_();layer.root.bias.zero_();layer.relation_weight.zero_()
            layer.relation_weight[0,0,0]=1.;layer.relation_weight[1,0,0]=2.
        out,_=layer(torch.tensor([[2.],[4.],[10.]]),torch.zeros(3,1),
                    torch.tensor([[0,1,0],[2,2,2]]),torch.tensor([0,0,1]))
        self.assertAlmostEqual(out[2].item(),17.,places=6)  # residual 10 + mean(2,4) + 2*2

    def test_masked_nan_never_enters_error_or_gradient(self):
        p=torch.tensor([[2.,float('nan')]],requires_grad=True)
        loss=masked_mse(p,torch.tensor([[1.,float('nan')]]),torch.tensor([[True,False]]))
        loss.backward();self.assertEqual(loss.item(),1.)
        torch.testing.assert_close(p.grad,torch.tensor([[2.,0.]]))

    def test_missing_fraction_and_weight_scale_invariance(self):
        a=masked_mse(torch.tensor([[2.,9.]]),torch.tensor([[1.,0.]]),torch.tensor([[True,False]]))
        b=masked_mse(torch.tensor([[2.,2.]]),torch.tensor([[1.,1.]]),torch.tensor([[True,True]]))
        self.assertEqual(a.item(),b.item())
        weighted=masked_mse(torch.tensor([[2.]]),torch.tensor([[1.]]),torch.tensor([[True]]),torch.tensor([.1]))
        self.assertAlmostEqual(weighted.item(),1.)

    def test_no_labels_produce_zero_gradient(self):
        p=torch.randn(2,3,requires_grad=True);loss=masked_mse(p,torch.full_like(p,float('nan')),torch.zeros_like(p,dtype=torch.bool))
        loss.backward();self.assertEqual(loss.item(),0.);self.assertTrue((p.grad==0).all())

    def test_symlog_roundtrip_preserves_negative_slew(self):
        values=torch.tensor([-1e9,-1.,0.,1e-4,1e8],dtype=torch.float64)
        torch.testing.assert_close(values,inverse_symlog(symlog(values)))

    def test_unknown_head_rejected_and_no_batchnorm(self):
        b=copy.copy(self.batch);b.group_ids=['unapproved']
        with self.assertRaises(ValueError):self.model(b)
        self.assertFalse(any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in self.model.modules()))


@unittest.skipUnless(HAS_TORCH,'Install requirements-gnn.txt to test model')
class RealAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        if not Path('data/fusion_v2/manifest.json').exists():raise unittest.SkipTest('Dataset not acquired')
        from src.dataset_fusion import FusionDataset
        from src.gnn.data import task_registry,input_schema,tensorize
        cls.dataset=FusionDataset();cls.tasks=task_registry(cls.dataset);cls.schema=input_schema(cls.dataset)
        cls.tensorize=staticmethod(tensorize)
        cls.row=next(cls.dataset.samples(topology='AICircuit__PA',materialize=False))
        torch.set_num_threads(1)

    def test_targets_do_not_enter_graph_inputs(self):
        other=copy.deepcopy(self.row)
        other['performance']=[v+1 if k else None for v,k in zip(other['performance'],other['mask'])]
        a=self.tensorize([self.row],self.dataset,self.tasks,self.schema)
        b=self.tensorize([other],self.dataset,self.tasks,self.schema)
        self.assertTrue(torch.equal(a.x,b.x));self.assertTrue(torch.equal(a.edge_attr,b.edge_attr))
        self.assertFalse(torch.equal(a.targets,b.targets))

    def test_mutual_coupling_parameters_reach_loss(self):
        torch.manual_seed(3)
        a=self.tensorize([self.row],self.dataset,self.tasks,self.schema)
        aux=a.edge_type>=self.schema['physical_relations'];self.assertTrue(aux.any())
        a.edge_attr.requires_grad_(True)
        model=EdgeCentricRGCN(self.schema,self.tasks,hidden_dim=8,num_layers=2,dropout=0.)
        loss,_=supervision_loss(model(a),a);loss.backward()
        self.assertGreater(a.edge_attr.grad[aux,:30].abs().sum().item(),0.)

    def test_smoke_predictor_rejects_production_use_and_untrained_metrics(self):
        from pathlib import Path
        from src.gnn.inference import load_predictor
        checkpoint=Path('artifacts/edge_rgcn_smoke/model.pt')
        if not checkpoint.exists():self.skipTest('Run check_edge_rgcn.py first')
        with self.assertRaisesRegex(ValueError,'smoke test'):load_predictor(checkpoint,self.dataset)
        predictor=load_predictor(checkpoint,self.dataset,allow_smoke=True)
        row=next(self.dataset.samples(topology='LOADBench__SimpleOpAmp1',materialize=False))
        result=predictor([row]);available=result['prediction_available']
        self.assertFalse(available[0,12].item());self.assertFalse(available[0,13].item())
        self.assertTrue(torch.isfinite(result['prediction'][available]).all())
        self.assertTrue(torch.isnan(result['prediction'][~available]).all())
        unseen=next(self.dataset.samples(topology='FALCON__ClassEPA',materialize=False))
        with self.assertRaisesRegex(ValueError,'never optimized'):predictor([unseen])


if __name__=='__main__':unittest.main()
