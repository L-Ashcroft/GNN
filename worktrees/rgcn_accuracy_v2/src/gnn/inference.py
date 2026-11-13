"""Checkpoint loading with explicit smoke/untrained-target checks."""
import hashlib
import torch
from .data import tensorize
from .model import EdgeCentricRGCN
from .losses import inverse_symlog


def load_predictor(path,dataset,allow_smoke=False):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['status']!='trained' and not allow_smoke:
        raise ValueError('This checkpoint is a construction smoke test, not a trained performance predictor')
    actual=hashlib.sha256((dataset.path/'manifest.json').read_bytes()).hexdigest()
    if actual!=checkpoint['dataset_manifest_sha256']:raise ValueError('Dataset graph/feature version differs from checkpoint')
    config=checkpoint['config']
    model=EdgeCentricRGCN(checkpoint['schema'],checkpoint['tasks'],
        **{k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges']},
        **{k:config.get(k,False) for k in ['use_domain_emb','use_film','collapse_relations']})
    model.load_state_dict(checkpoint['model']);model.eval()
    return Predictor(model,dataset,set(checkpoint['trained_group_ids']),checkpoint['trained_target_slots'],
                     set(checkpoint.get('trained_domain_ids',[])))


class Predictor:
    def __init__(self,model,dataset,trained_groups,trained_slots,trained_domains=None):
        self.model=model;self.dataset=dataset;self.trained_groups=trained_groups;self.trained_slots=trained_slots
        self.trained_domains=trained_domains or set()

    @torch.no_grad()
    def __call__(self,rows):
        batch=tensorize(rows,self.dataset,self.model.tasks,self.model.schema)
        missing=set(batch.group_ids)-self.trained_groups
        if missing:raise ValueError('Target heads were never optimized: '+','.join(sorted(missing)))
        if self.model.use_domain_emb and not set(batch.domain_ids.tolist())<=self.trained_domains:
            raise ValueError('Domain embedding never optimized')
        transformed=self.model(batch)
        available=torch.tensor([[i in self.trained_slots[group] for i in range(16)] for group in batch.group_ids],dtype=torch.bool)
        prediction=inverse_symlog(transformed)*batch.scales.to(torch.float64)
        return {'prediction':prediction.masked_fill(~available,float('nan')),'prediction_available':available,
                'target_semantic_ids':batch.semantic_ids,'observed_target_mask':batch.mask,
                'sample_ids':batch.sample_ids,'group_ids':batch.group_ids}
