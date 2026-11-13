import ast
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest

from src.graph.netlist import read_logical_lines,spice_number
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from src.graph.builder import CircuitBatch,sanity_check
    from src.model.network import EdgeCentricRGCN
    from src.model.film import FiLM
    from src.train.losses import symlog
    from src.train.metrics import MetricAccumulator
    from src.train.stage import build_optimizer,parameter_counts
    from src.train.loop import TrainingCoverage,EarlyStopping,train_epoch,evaluate,fit


class GraphModuleTests(unittest.TestCase):
    def test_spice_units_not_spectre_units(self):
        self.assertEqual(spice_number('2M'),.002);self.assertEqual(spice_number('2meg'),2e6)
        self.assertAlmostEqual(spice_number('1.2u'),1.2e-6)
        with self.assertRaises(ValueError):spice_number('2bad')

    def test_continuation_and_orphan(self):
        Path('.cache/tmp').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir='.cache/tmp') as folder:
            p=Path(folder)/'sample.cir';p.write_text('M0 d g s b model\n+ w=2u l=1u\n*comment\nR0 a b 1k\n')
            self.assertEqual(len(read_logical_lines(p)),2);self.assertIn('w=2u',read_logical_lines(p)[0])
            p.write_text('+ w=2u\n')
            with self.assertRaises(ValueError):read_logical_lines(p)

    def test_graph_directory_has_no_nn_import(self):
        for path in Path('src/graph').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,['torch.nn','src.model'])
                if isinstance(node,ast.Import):self.assertFalse(any(n.name.startswith('torch.nn') for n in node.names))


@unittest.skipUnless(HAS_TORCH,'Install GNN requirements')
class Lecture5Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9);torch.set_num_threads(1)
        self.schema={'node_features':['a','b'],'device_features':['w'],'relations':['D>G','G>D'],
                     'domains':[{'id':0,'source_dataset':'A','process':'unknown'},{'id':1,'source_dataset':'B','process':'known'}]}
        self.tasks={'task':{'slots':{'0':{'family':'power','semantic_id':'task::power','unit':'W','original_name':'p','scale':1.}}}}
        self.model=EdgeCentricRGCN(self.schema,self.tasks,hidden_dim=8,num_layers=2,dropout=0.,use_domain_emb=True,use_film=True)
        self.batch=CircuitBatch(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([[0,1],[1,0]]),torch.randn(2,3),
            torch.tensor([0,1]),torch.tensor([0,0]),torch.tensor([[1.]+[0.]*15]),torch.tensor([[True]+[False]*15]),
            torch.ones(1,16),['task'],[['task::power']+[None]*15],['id0'],torch.tensor([0]),'train')

    def test_film_identity_matches_disabled_branch(self):
        e=torch.randn(3,8);gamma,beta=FiLM(8,8)(e)
        self.assertTrue((gamma==1).all());self.assertTrue((beta==0).all())
        self.model.eval();a=self.model(self.batch);self.model.use_film=False
        torch.testing.assert_close(a,self.model(self.batch),rtol=0,atol=0)

    def test_domain_broadcast_batch_and_gradients(self):
        result=self.model(self.batch);result[:,0].sum().backward()
        self.assertGreater(self.model.films[0].to_gamma_beta.weight.grad.abs().sum(),0.)
        self.assertGreater(self.model.domain_emb.embedding.weight.grad[0].abs().sum(),0.)
        self.assertEqual(self.model.domain_emb.embedding.weight.grad[1].abs().sum(),0.)
        self.batch.domain_ids=torch.tensor([2])
        with self.assertRaises(ValueError):self.model(self.batch)

    def test_actual_modules_freeze_and_stage_reset(self):
        all_count=parameter_counts(self.model)['total'];optimizer=build_optimizer(self.model,'finetune_head')
        self.assertTrue(all(not p.requires_grad for p in self.model.layers.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.films.parameters()))
        self.assertLess(parameter_counts(self.model)['trainable'],all_count)
        optimizer=build_optimizer(self.model,'finetune_all')
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[2e-5,2e-4])
        build_optimizer(self.model,'pretrain');self.assertEqual(parameter_counts(self.model)['trainable'],all_count)

    def test_validation_cannot_train_and_evaluate_does_not_modify_model(self):
        bad=copy.copy(self.batch);bad.split_role='validation';optimizer=build_optimizer(self.model)
        with self.assertRaises(ValueError):train_epoch(self.model,[bad],optimizer,TrainingCoverage())
        coverage=TrainingCoverage();coverage.observe(self.batch)
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        evaluate(self.model,[bad],coverage)
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_early_stopping_restores_copied_best_weights(self):
        stop=EarlyStopping(patience=1);stop.step(.5,self.model,0)
        best={k:v.clone() for k,v in self.model.state_dict().items()}
        with torch.no_grad():next(self.model.parameters()).add_(1.)
        self.assertTrue(stop.step(.4,self.model,1));stop.restore(self.model)
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,best[k]))

    def test_loss_metric_keys_cannot_mix_units_or_semantics(self):
        b=copy.copy(self.batch);b.group_ids=['task','task'];b.targets=torch.tensor([[1.]+[0.]*15,[2.]+[0.]*15])
        b.mask=self.batch.mask.repeat(2,1);b.scales=torch.ones(2,16)
        m=MetricAccumulator(self.tasks);m.update(symlog(b.targets/b.scales),b);r=m.compute()
        self.assertAlmostEqual(r['macro_r2'],1.);self.assertAlmostEqual(r['macro_hit5'],1.)
        a=MetricAccumulator(self.tasks);a.update(symlog(self.batch.targets),self.batch)
        self.assertIsNone(a.compute()['per_target'][0]['r2'])
        tasks=copy.deepcopy(self.tasks);tasks['task']['slots']['0']['unit']='dB'
        d=MetricAccumulator(tasks);d.update(symlog(b.targets),b)
        self.assertIsNone(d.compute()['macro_hit5']);self.assertIsNone(d.compute()['macro_mre'])

    def test_collapsed_relations_and_shape_assertion(self):
        model=EdgeCentricRGCN(self.schema,self.tasks,hidden_dim=8,num_layers=1,collapse_relations=True)
        self.assertEqual(model.layers[0].num_relations,1);self.assertEqual(tuple(model(self.batch).shape),(1,16))
        self.assertEqual(sanity_check(self.batch,2)['message_edges'],2)
        bad=copy.copy(self.batch);bad.edge_index=torch.tensor([[0,2],[1,0]])
        with self.assertRaises(ValueError):sanity_check(bad,2)

    def test_coverage_rejects_untrained_slot_or_domain(self):
        coverage=TrainingCoverage();coverage.observe(self.batch)
        b=copy.copy(self.batch);b.domain_ids=torch.tensor([1])
        with self.assertRaises(ValueError):coverage.check(b,True)
        b=copy.copy(self.batch);b.mask=b.mask.clone();b.mask[0,1]=True
        with self.assertRaises(ValueError):coverage.check(b)

    def test_film_checkpoint_roundtrip_through_predictor(self):
        checkpoint=Path('artifacts/lecture5_smoke/model.pt')
        if not checkpoint.exists():self.skipTest('Run check_lecture5.py first')
        from src.dataset_fusion import FusionDataset
        from src.gnn.inference import load_predictor
        dataset=FusionDataset()
        with self.assertRaises(ValueError):load_predictor(checkpoint,dataset)
        predictor=load_predictor(checkpoint,dataset,allow_smoke=True)
        self.assertTrue(predictor.model.use_film)
        row=next(dataset.samples(topology='LOADBench__SimpleOpAmp1',materialize=False))
        result=predictor([row]);known=result['prediction_available']
        self.assertTrue(torch.isfinite(result['prediction'][known]).all())
        self.assertTrue(torch.isnan(result['prediction'][~known]).all())
