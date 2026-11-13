"""Edge cases and ablation compatibility, against the frozen original code."""
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];sys.path.insert(0,str(ROOT))
from src.model.execution import prepare_batch,move_batch
from src.model.network import EdgeCentricRGCN
from src.model.layers import EdgeRGCNLayer,AttentionPool
from reference_src.model.network import EdgeCentricRGCN as Original
from reference_src.model.layers import EdgeRGCNLayer as OriginalLayer,AttentionPool as OriginalPool
from src.train.losses import supervision_loss
from reference_src.train.losses import supervision_loss as original_loss
from src.train.loop import TrainingCoverage,LossAccumulator
from reference_src.train.loop import TrainingCoverage as OriginalCoverage,LossAccumulator as OriginalAccumulator

class Batch(SimpleNamespace):
    @property
    def num_graphs(self):return len(self.group_ids)
    def to(self,device):return move_batch(self,device)

def read(path):return json.loads(path.read_text(encoding='utf-8'))

class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.fields=torch.load(WORKSPACE/'reports/experiment1_performance/fixtures/batch_16_7.pt',weights_only=True)
        config=read(WORKSPACE/'config/experiment1_fusion.json');cache=WORKSPACE/config['cache']
        cls.schema=read(cache/'schema.json');cls.tasks=read(cache/'tasks.json')
        cls.kwargs={k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}

    def batch(self):return Batch(**copy.deepcopy(self.fields))

    def test_plan_reuse_and_mask_invalidation(self):
        batch=self.batch();first=prepare_batch(batch);self.assertIs(first,prepare_batch(batch))
        batch.mask[0,0]=~batch.mask[0,0];self.assertIsNot(first,prepare_batch(batch))
        self.assertTrue(torch.equal(batch.to('cpu').execution_plan.tensors['relation_order'],prepare_batch(batch).tensors['relation_order']))

    def test_cached_relation_degrees_keep_typed_duplicates_and_loops(self):
        torch.manual_seed(7)
        edges=torch.tensor([[0,0,0,1,1,2],[1,1,1,1,2,2]])
        types=torch.tensor([0,0,1,1,47,47]);h=torch.randn(3,4);z=torch.randn(6,4)
        batch=Batch(x=h,edge_index=edges,edge_type=types,graph_index=torch.zeros(3,dtype=torch.long),
                    group_ids=['x'],mask=torch.ones(1,16,dtype=torch.bool),domain_ids=None)
        left=OriginalLayer(4,48,dropout=0.);right=EdgeRGCNLayer(4,48,dropout=0.);right.load_state_dict(left.state_dict())
        a=left(h,z,edges,types);b=right(h,z,edges,types,relation_plan=prepare_batch(batch).relations())
        for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=0,rtol=0)
        plan=prepare_batch(batch);order=plan.tensors['relation_order']
        self.assertEqual(plan.tensors['relation_count'][order==0].item(),2)

    def test_bucket_pool_noncontiguous_and_empty_graphs(self):
        torch.manual_seed(9);left=OriginalPool(8);right=AttentionPool(8);right.load_state_dict(left.state_dict())
        graph=torch.tensor([1,0,1,0,2]);h=torch.randn(5,8,requires_grad=True);j=h.detach().clone().requires_grad_()
        a=left(h,graph,5);b=right(j,graph,5)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=1e-6,rtol=1e-5)
        (a[0].square().sum()+a[1].square().sum()).backward();(b[0].square().sum()+b[1].square().sum()).backward()
        torch.testing.assert_close(h.grad,j.grad,atol=1e-6,rtol=1e-5)
        empty=torch.empty(0,8,requires_grad=True);g=torch.empty(0,dtype=torch.long)
        left.zero_grad(set_to_none=True);right.zero_grad(set_to_none=True)
        left(empty,g,2)[0].sum().backward();right(empty,g,2)[0].sum().backward()
        for p,q in zip(left.parameters(),right.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_missing_nan_loss_and_gradient(self):
        batch=self.batch();torch.manual_seed(4);p=torch.randn(batch.mask.shape)
        p[~batch.mask]=float('nan');batch.targets[~batch.mask]=float('nan')
        a=p.clone().requires_grad_();b=p.clone().requires_grad_()
        la,ga=original_loss(a,batch);lb,gb=supervision_loss(b,batch)
        torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-5);self.assertEqual(list(ga),list(gb))
        la.backward();lb.backward();torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
        self.assertTrue(torch.isfinite(b.grad).all());self.assertTrue((b.grad[~batch.mask]==0).all())

    def test_no_labels_and_observed_nonfinite_rejected(self):
        batch=self.batch();prediction=torch.zeros_like(batch.targets);batch.mask.zero_()
        for function in [original_loss,supervision_loss]:
            with self.assertRaises(ValueError):function(prediction,batch)
        batch=self.batch();row,column=torch.nonzero(batch.mask,as_tuple=True);prediction[row[0],column[0]]=float('nan')
        for function in [original_loss,supervision_loss]:
            with self.assertRaises(ValueError):function(prediction,batch)

    def test_zero_observation_group_coverage_and_accumulator(self):
        batch=self.batch();removed=batch.group_ids[0]
        for i,g in enumerate(batch.group_ids):
            if g==removed:batch.mask[i].zero_()
        prediction=torch.zeros_like(batch.targets)
        a,ga=original_loss(prediction,batch);b,gb=supervision_loss(prediction,batch)
        self.assertNotIn(removed,ga);self.assertEqual(list(ga),list(gb));torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        left=OriginalCoverage();right=TrainingCoverage();left.observe(batch);right.observe(batch)
        self.assertEqual(left.slots,right.slots);self.assertEqual(left.domains,right.domains)
        right.check(batch,True)
        for cls in [OriginalAccumulator,LossAccumulator]:
            total=cls();total.update(prediction,batch);self.assertNotIn(removed,total.values)
        unknown=self.batch();unknown.domain_ids.fill_(999)
        with self.assertRaises(ValueError):right.check(unknown,True)

    def test_pool_membership_permutation_and_cache_rebuild(self):
        batch=self.batch();prepare_batch(batch);old_plan=batch.execution_plan;torch.manual_seed(1)
        order=torch.randperm(len(batch.x));inverse=torch.argsort(order)
        batch.x=batch.x[order];batch.graph_index=batch.graph_index[order];batch.edge_index=inverse[batch.edge_index]
        edge_order=torch.randperm(len(batch.edge_type));batch.edge_index=batch.edge_index[:,edge_order]
        batch.edge_type=batch.edge_type[edge_order];batch.edge_attr=batch.edge_attr[edge_order]
        torch.manual_seed(3);left=Original(self.schema,self.tasks,**self.kwargs);right=EdgeCentricRGCN(self.schema,self.tasks,**self.kwargs)
        right.load_state_dict(left.state_dict());left.eval();right.eval()
        with torch.no_grad():torch.testing.assert_close(left(batch),right(batch),atol=5e-6,rtol=5e-5)
        self.assertIsNot(batch.execution_plan,old_plan)

    def test_supported_ablations_forward_backward_and_rng(self):
        variants=[{'collapse_relations':True},{'update_edges':False},{'use_domain_emb':False,'use_film':False}]
        for change in variants:
            with self.subTest(change=change):
                kwargs=dict(self.kwargs,**change);torch.manual_seed(22)
                left=Original(self.schema,self.tasks,**kwargs);right=EdgeCentricRGCN(self.schema,self.tasks,**kwargs)
                right.load_state_dict(left.state_dict());batch=self.batch();rng=torch.get_rng_state()
                a=left(batch,True);la,_=original_loss(a['prediction'],batch);la.backward();end=torch.get_rng_state()
                torch.set_rng_state(rng);b=right(batch,True);lb,_=supervision_loss(b['prediction'],batch);lb.backward()
                self.assertTrue(torch.equal(end,torch.get_rng_state()))
                for name in a:torch.testing.assert_close(a[name],b[name],atol=5e-6,rtol=5e-5)
                for (name,p),(other,q) in zip(left.named_parameters(),right.named_parameters()):
                    self.assertEqual(name,other);self.assertEqual(p.grad is None,q.grad is None)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-3)

    def test_checkpoint_and_optimizer_state_compatibility(self):
        from reference_src.train.stage import build_optimizer as old_optimizer
        from src.train.stage import build_optimizer
        torch.manual_seed(6);left=Original(self.schema,self.tasks,**self.kwargs);right=EdgeCentricRGCN(self.schema,self.tasks,**self.kwargs)
        right.load_state_dict(left.state_dict(),strict=True);left.load_state_dict(right.state_dict(),strict=True)
        self.assertEqual([(n,p.shape) for n,p in left.named_parameters()],[(n,p.shape) for n,p in right.named_parameters()])
        old=old_optimizer(left);new=build_optimizer(right)
        batch=self.batch();loss,_=original_loss(left(batch),batch);loss.backward();old.step()
        new.load_state_dict(copy.deepcopy(old.state_dict()));self.assertEqual(old.param_groups[0]['name'],new.param_groups[0]['name'])
        self.assertEqual(len(old.state),len(new.state))

    def test_grouped_heads_real_batch256(self):
        from reference_src.model.heads import SemanticGroupedHead as OldHead
        from src.model.heads import SemanticGroupedHead as NewHead
        batch=Batch(**torch.load(WORKSPACE/'reports/experiment1_performance/fixtures/batch_256_11.pt',weights_only=True))
        torch.manual_seed(55);left=OldHead(128,self.tasks);right=NewHead(128,self.tasks);right.load_state_dict(left.state_dict())
        h=torch.randn(256,128,requires_grad=True);j=h.detach().clone().requires_grad_()
        a=left(h,batch.group_ids);b=right(j,batch.group_ids,prepare_batch(batch).grouping())
        torch.testing.assert_close(a,b,atol=5e-6,rtol=5e-5)
        la,ga=original_loss(a,batch);lb,gb=supervision_loss(b,batch)
        self.assertEqual(list(ga),list(gb));torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-5)
        la.backward();lb.backward();torch.testing.assert_close(h.grad,j.grad,atol=2e-5,rtol=2e-3)
        maximum=0.
        for p,q in zip(left.parameters(),right.parameters()):
            self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-3)
                maximum=max(maximum,(p.grad-q.grad).abs().max().item())
        (ROOT/'results/head_batch256.json').write_text(json.dumps({'batch_size':256,'contracts':len(set(batch.group_ids)),
            'prediction_max_abs':(a-b).abs().max().item(),'head_gradient_max_abs':maximum,'status':'passed',
            'scope':'CPU isolated head/loss; real grouping and labels, synthetic embedding unit-test input; no full GNN'}),encoding='utf-8')

    def test_pooling_real_batch256_structure(self):
        batch=Batch(**torch.load(WORKSPACE/'reports/experiment1_performance/fixtures/batch_256_11.pt',weights_only=True))
        plan=prepare_batch(batch);checks=[]
        for kind,membership in [('node',batch.graph_index),('edge',batch.graph_index[batch.edge_index[0]])]:
            torch.manual_seed(77);left=OriginalPool(128);right=AttentionPool(128);right.load_state_dict(left.state_dict())
            h=torch.randn(len(membership),128,requires_grad=True);j=h.detach().clone().requires_grad_()
            a=left(h,membership,256);b=right(j,membership,256,plan.pooling(kind))
            for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=5e-6,rtol=5e-5)
            (a[0].square().mean()+a[1].square().mean()).backward()
            (b[0].square().mean()+b[1].square().mean()).backward()
            torch.testing.assert_close(h.grad,j.grad,atol=2e-5,rtol=2e-3)
            for p,q in zip(left.parameters(),right.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-3)
            checks.append({'kind':kind,'buckets':len(plan.pool_buckets[kind]),'pooled_max_abs':(a[0]-b[0]).abs().max().item(),
                'attention_max_abs':(a[1]-b[1]).abs().max().item(),'input_gradient_max_abs':(h.grad-j.grad).abs().max().item()})
        (ROOT/'results/pooling_batch256.json').write_text(json.dumps({'status':'passed','checks':checks,
            'scope':'CPU isolated pooling on real graph memberships, synthetic hidden-state unit-test input; no full GNN'}),encoding='utf-8')

if __name__=='__main__':unittest.main(verbosity=2)
