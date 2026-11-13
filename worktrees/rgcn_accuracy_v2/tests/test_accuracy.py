"""CPU correctness/engineering tests. Toy data exists only inside unit tests."""
import copy
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import psutil
import torch
from src.graph.builder import CircuitBatch
from src.model.heads import SemanticGroupedHead
from src.model.contract_head import ContractResidualHead
from src.model.network import EdgeCentricRGCN
from src.model.accuracy_network import AccuracyRGCN
from src.model.execution import prepare_batch
from src.train.target_calibration import fit,validate,sha
from src.train.accuracy_loss import supervision_loss
from src.train.losses import symlog,inverse_symlog
from src.train.stage import build_optimizer
from src.train.accuracy_loop import LossAccumulator

PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1]
read=lambda path:json.loads(path.read_text(encoding='utf-8'))

class AccuracyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.fields=torch.load(ROOT/'reports/experiment1_performance/fixtures/batch_16_7.pt',map_location='cpu',weights_only=True)
        cache=ROOT/'data/model_ready/fusion_v2_training_v1'
        cls.schema=read(cache/'schema.json');cls.tasks=read(cache/'tasks.json');cls.manifest=read(cache/'manifest.json')
        cls.calibration=read(ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json')
        cls.kwargs={k:read(ROOT/'config/experiment1_fusion.json')[k] for k in
            ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}

    def tearDown(self):gc.collect()
    def batch(self):return CircuitBatch(**copy.deepcopy(self.fields))
    def head(self,**kwargs):return ContractResidualHead(128,self.tasks,self.calibration,**kwargs)

    def test_artifact_train_only_and_semantics(self):
        validate(self.calibration,self.tasks,self.manifest)
        self.assertEqual(self.calibration['role_counts']['train'],400083)
        damaged=copy.deepcopy(self.calibration);damaged['split_sha256']='changed'
        with self.assertRaises(ValueError):validate(damaged,self.tasks,self.manifest)
        damaged=copy.deepcopy(self.calibration);g=next(iter(damaged['groups']));s=next(iter(damaged['groups'][g]['slots']))
        damaged['groups'][g]['slots'][s]['semantic_id']='wrong_label'
        with self.assertRaises(ValueError):validate(damaged,self.tasks,self.manifest)

    def test_fitting_cannot_use_heldout_labels(self):
        # Explicitly artificial fixture, never written to a dataset or report.
        with tempfile.TemporaryDirectory(dir=PACKAGE/'results',prefix='unit_calibration_') as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(ROOT));cache=root/'cache';cache.mkdir();folder=root/'topology';folder.mkdir()
            (root/'split').write_text('unit-test split');task={'g':{'slots':{'0':{'semantic_id':'unit::y','scale':1.}}}}
            (cache/'tasks.json').write_text(json.dumps(task));np.save(folder/'role.npy',np.array([0,0,1,2]))
            np.save(folder/'group.npy',np.zeros(4,dtype=np.int16));np.save(folder/'sample_ids.npy',np.array([b'a',b'b',b'c',b'd']))
            mask=np.zeros((4,16),dtype=bool);mask[:,0]=True;np.save(folder/'mask.npy',mask)
            def write_manifest():
                m={'group_names':['g'],'tasks_sha256':sha(cache/'tasks.json'),'split_sha256':sha(root/'split'),
                   'role_counts':{'train':2,'validation':1,'test':1},
                   'topologies':{'t':{'folder':'topology','cache_sha256':{p.name:sha(p) for p in folder.glob('*.npy')}}}}
                task['g']['contract']={'source_dataset':'unit'};(cache/'tasks.json').write_text(json.dumps(task));m['tasks_sha256']=sha(cache/'tasks.json')
                (cache/'manifest.json').write_text(json.dumps(m))
            values=np.zeros((4,16),dtype=np.float32);values[:,0]=[1,3,20,40];np.save(folder/'target.npy',values);write_manifest()
            first=fit(root,'cache','split',root/'one.json')
            values[2:,0]=[-1e9,1e9];np.save(folder/'target.npy',values);write_manifest()
            second=fit(root,'cache','split',root/'two.json')
            self.assertEqual(first['groups'],second['groups']);self.assertEqual(first['training_sample_ids_sha256'],second['training_sample_ids_sha256'])
            self.assertFalse(first['validation_labels_used']);self.assertFalse(first['test_labels_used'])

    def test_zero_residual_preserves_legacy_head(self):
        batch=self.batch();torch.manual_seed(7);base=SemanticGroupedHead(128,self.tasks)
        candidate=ContractResidualHead(128,self.tasks,self.calibration,base=base,calibrated_targets=False)
        embedding=torch.randn(batch.num_graphs,128)
        torch.testing.assert_close(base(embedding,batch.group_ids),candidate(embedding,batch.group_ids,prepare_batch(batch).grouping()),atol=0,rtol=0)
        self.assertEqual(len(candidate.adapters),137)

    def test_contract_residual_does_not_modify_other_contracts(self):
        head=self.head(calibrated_targets=False,adapter_min_train_samples=0)
        groups=sorted(set(self.fields['group_ids']))[:2];self.assertEqual(len(groups),2);embedding=torch.randn(2,128)
        a=head(embedding,groups).detach()
        with torch.no_grad():head.adapters[groups[0]][-1].bias.fill_(.25)
        b=head(embedding,groups).detach()
        torch.testing.assert_close(a[1],b[1],atol=0,rtol=0);self.assertGreater(float((a[0]-b[0]).abs().max()),0)

    def test_grouped_head_cpu_predictions_loss_and_gradients(self):
        from src.model.grouped_contract_head import forward_grouped
        batch=self.batch();left=self.head();right=copy.deepcopy(left);x=torch.randn(batch.num_graphs,128)
        a=left(x,batch.group_ids);b=forward_grouped(right,x,batch.group_ids,prepare_batch(batch).grouping())
        torch.testing.assert_close(a,b,atol=5e-6,rtol=5e-5)
        la,_=supervision_loss(a,batch,left);lb,_=supervision_loss(b,batch,right)
        torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-5);la.backward();lb.backward()
        for (name,p),(other,q) in zip(left.named_parameters(),right.named_parameters()):
            self.assertEqual(name,other);self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-3)

    def test_masked_normalized_loss_formula_and_gradients(self):
        batch=self.batch();head=self.head();prediction=torch.randn_like(batch.targets,requires_grad=True)
        with torch.no_grad():prediction[~batch.mask]=float('nan');batch.targets[~batch.mask]=float('nan')
        loss,groups=supervision_loss(prediction,batch,head)
        target=symlog(batch.targets/batch.scales);_,std=head.affine(batch.group_ids);expected=[]
        for group in sorted(set(batch.group_ids)):
            rows=torch.tensor([i for i,g in enumerate(batch.group_ids) if g==group]);valid=batch.mask[rows]
            if valid.any():expected.append(((prediction[rows][valid]-target[rows][valid])/std[rows][valid]).square().mean())
        torch.testing.assert_close(loss,torch.stack(expected).mean(),atol=1e-6,rtol=1e-6)
        loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all());self.assertTrue((prediction.grad[~batch.mask]==0).all())
        accumulator=LossAccumulator(head);accumulator.update(prediction.detach(),batch)
        self.assertEqual(set(accumulator.values),set(groups))
        torch.testing.assert_close(torch.tensor(accumulator.mean()),loss.detach(),atol=1e-6,rtol=1e-6)

    def test_observed_nonfinite_empty_and_untrained_slots_rejected(self):
        batch=self.batch();head=self.head();prediction=torch.zeros_like(batch.targets)
        row,column=torch.nonzero(batch.mask,as_tuple=True)
        prediction[row[0],column[0]]=float('inf')
        with self.assertRaises(ValueError):supervision_loss(prediction,batch,head)
        batch.mask.zero_()
        with self.assertRaises(ValueError):supervision_loss(prediction,batch,head)
        unseen=next((g,int(s)) for g,r in self.calibration['groups'].items() for s,v in r['slots'].items() if not v['count'])
        with self.assertRaises(ValueError):head.check_observed({unseen[0]:[unseen[1]]})
        with self.assertRaises(ValueError):head(torch.zeros(1,128),['unknown_contract'])

    def test_affine_roundtrip_and_observed_constant_slots_retained(self):
        batch=self.batch();head=self.head(contract_residual=False)
        for blocks in head.base.readouts.values():
            for layer in blocks.values():torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        prediction=head(torch.randn(batch.num_graphs,128),batch.group_ids);center,std=head.affine(batch.group_ids)
        torch.testing.assert_close(prediction,center,atol=0,rtol=0)
        self.assertTrue((std>0).all());self.assertTrue(torch.isfinite(inverse_symlog(prediction)).all())
        observed=sum(s['count']>0 for g in self.calibration['groups'].values() for s in g['slots'].values())
        self.assertEqual(observed,1491)

    def test_state_dict_buffers_and_strict_own_checkpoint(self):
        head=self.head();buffer=io.BytesIO();torch.save(head.state_dict(),buffer);buffer.seek(0)
        other=self.head();other.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        batch=self.batch();x=torch.randn(batch.num_graphs,128)
        torch.testing.assert_close(head(x,batch.group_ids),other(x,batch.group_ids),atol=0,rtol=0)
        self.assertIn('target_center',head.state_dict());self.assertIn('target_std',head.state_dict())

    def test_new_inference_checkpoint_and_physical_units(self):
        from src.gnn.accuracy_inference import load_model,predict_physical
        config=dict(self.kwargs,contract_residual=True,calibrated_targets=True,adapter_hidden=32,adapter_min_train_samples=256)
        model=AccuracyRGCN(self.schema,self.tasks,self.calibration,**config);model.eval();batch=self.batch()
        with tempfile.TemporaryDirectory(dir=PACKAGE/'results',prefix='unit_checkpoint_') as directory:
            path=Path(directory).resolve()/'unit.pt';self.assertTrue(path.is_relative_to(ROOT))
            torch.save({'status':'trained','model_class':'AccuracyRGCN','config':config,'target_calibration':self.calibration,
                        'schema':self.schema,'tasks':self.tasks,'model':model.state_dict()},path)
            restored,_=load_model(path)
            a=predict_physical(model,batch);b=predict_physical(restored,batch)
            torch.testing.assert_close(a['prediction'],b['prediction'],atol=0,rtol=0)
            self.assertTrue(torch.equal(b['mask'],batch.mask));self.assertEqual(b['sample_ids'],batch.sample_ids)
            torch.save({'status':'trained','model_class':'EdgeCentricRGCN'},path)
            with self.assertRaises(ValueError):load_model(path)

    def test_validation_loop_keeps_physical_metrics_and_counts(self):
        from src.train.accuracy_loop import evaluate,TrainingCoverage
        # Cheap real-head model isolates evaluation routing/metric inversion.
        class HeadOnly(torch.nn.Module):
            def __init__(self,tasks,head):
                super().__init__();self.tasks=tasks;self.head=head;self.use_domain_emb=False
            def forward(self,batch):return self.head(torch.zeros(batch.num_graphs,128),batch.group_ids)
        batch=self.batch();batch.split_role='validation';head=self.head();model=HeadOnly(self.tasks,head)
        coverage=TrainingCoverage();coverage.observe(batch)
        result=evaluate(model,[batch],coverage)
        self.assertEqual(result['samples'],batch.num_graphs);self.assertGreater(result['target_groups'],0)
        self.assertEqual(result['loss_space'],'train-standardized symlog')
        self.assertTrue(all(np.isfinite(row['mae']) for row in result['per_target']))

    def test_full_backbone_real_batch_and_cpu_optimizer_steps(self):
        if psutil.virtual_memory().available<768*2**20:self.skipTest('RAM guard: defer full-backbone CPU test')
        original=self.batch();fields=vars(original).copy();keep=original.graph_index<4;nodes=torch.nonzero(keep).flatten()
        mapping=torch.full((len(original.x),),-1,dtype=torch.long);mapping[nodes]=torch.arange(len(nodes))
        edges=keep[original.edge_index[0]]
        for key in ['x','graph_index']:fields[key]=fields[key][keep]
        fields['edge_index']=mapping[original.edge_index[:,edges]]
        for key in ['edge_attr','edge_type']:fields[key]=fields[key][edges]
        for key in ['targets','mask','scales','domain_ids']:fields[key]=fields[key][:4]
        for key in ['group_ids','semantic_ids','sample_ids']:fields[key]=fields[key][:4]
        fields['execution_plan']=None;batch=CircuitBatch(**fields)
        torch.manual_seed(42);legacy=EdgeCentricRGCN(self.schema,self.tasks,**self.kwargs)
        model=AccuracyRGCN(self.schema,self.tasks,self.calibration,calibrated_targets=False,**self.kwargs)
        model.initialize_from_legacy(legacy.state_dict());legacy.eval();model.eval()
        with torch.no_grad():
            a=legacy(batch,True);b=model(batch,True)
            for key in a:torch.testing.assert_close(a[key],b[key],atol=5e-6,rtol=5e-5)
        del legacy,model,a,b;gc.collect()
        model=AccuracyRGCN(self.schema,self.tasks,self.calibration,**self.kwargs);model.eval()
        with self.assertRaises(ValueError):model.initialize_from_legacy({})
        optimizer=build_optimizer(model,lr_head=2e-4,weight_decay=.01);losses=[]
        # Fixed four real training graphs. Engineering overfit check, not a
        # validation-quality estimate; eval disables Dropout but retains grads.
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True);prediction=model(batch);loss,_=supervision_loss(prediction,batch,model.head)
            self.assertTrue(torch.isfinite(loss));losses.append(float(loss.detach()));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        self.assertLess(min(losses[1:]),losses[0])
        model.train();optimizer.zero_grad(set_to_none=True)
        training_loss,_=supervision_loss(model(batch),batch,model.head);training_loss.backward()
        self.assertTrue(torch.isfinite(training_loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        result={'scope':'CPU four real training graphs, full 4-layer 128-dimension 48-relation model; eval-mode engineering fit only',
                'losses':losses,'parameters':sum(p.numel() for p in model.parameters()),'sample_ids':batch.sample_ids,
                'no_validation_quality_claim':True,'GPU_tested':False,'backbone_legacy_states_equal':True,
                'additional_train_mode_dropout_AdamW_step_passed':True,'train_mode_loss':float(training_loss.detach())}
        (PACKAGE/'results/cpu_tiny_fit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':unittest.main()
