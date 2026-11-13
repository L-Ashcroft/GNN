import unittest
import numpy as np
import pandas as pd
import torch
from src.common import ROOT,read
from scripts.split_fusion_stratified import make_split
from src.train.fusion_cache import vector_evaluate,CachedFusion
from src.netlist import evaluate
from src.dataset_fusion import FusionDataset
from src.graph.builder import tensorize,task_registry,input_schema


class Experiment1Tests(unittest.TestCase):
    def test_grouped_stratification_is_reproducible_and_label_free(self):
        rows=[]
        for source in ['A','B']:
            for i in range(30):rows.append({'sample_id':source+str(i),'design_group':str(i) if i<3 else source+str(i),
                'source_dataset':source,'circuit_family':'amp','connectivity_group':'same','topology_id':source,
                'parameter_stratum':i%3,'performance':float(i)})
        index=pd.DataFrame(rows);a,_,_=make_split(index);b,_,_=make_split(index.sample(frac=1,random_state=9))
        self.assertEqual(a.set_index('sample_id').role.to_dict(),b.set_index('sample_id').role.to_dict())
        self.assertEqual(a.groupby('design_group').role.nunique().max(),1)
        self.assertEqual(set(a.role),{'train','validation','test'})

    def test_vector_expressions_match_scalar(self):
        env={'w':np.array([1.,2.,3.]),'l':np.array([.1,.2,.3])}
        for text in ['w/l','sqrt(w*l)+1u','-(w**2)+l']:
            actual=vector_evaluate(text,env)
            expected=[evaluate(text,{k:v[i] for k,v in env.items()}) for i in range(3)]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_cached_mixed_batch_matches_original(self):
        config=read(ROOT/'config/experiment1_fusion.json')
        if not (ROOT/config['cache']/'manifest.json').exists():self.skipTest('Cache not prepared')
        cached=CachedFusion(config);dataset=FusionDataset();rows=[];references=[]
        for name in ['FALCON__CGVA','AICircuit__Transmitter','LOADBench__SimpleOpAmp1']:
            import pyarrow.parquet as pq
            row=pq.read_table(dataset.path/'sampled'/(name+'.parquet')).slice(0,1).to_pylist()[0]
            rows.append(row);references.append((name,0))
        actual=cached.batch(references,'train');expected=tensorize(rows,dataset,task_registry(dataset),input_schema(dataset))
        for field in ['x','edge_index','edge_type','edge_attr','targets','mask','scales','domain_ids']:
            torch.testing.assert_close(getattr(actual,field),getattr(expected,field),rtol=2e-6,atol=2e-6)
        self.assertEqual(actual.sample_ids,expected.sample_ids);self.assertEqual(actual.group_ids,expected.group_ids)


if __name__=='__main__':unittest.main()
