"""Explicit loader for the new head/calibration format; not the legacy loader."""
from pathlib import Path
import torch
from ..model.accuracy_network import AccuracyRGCN
from ..train.losses import inverse_symlog

KEYS=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations',
      'contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']

def load_model(path,device='cpu'):
    saved=torch.load(Path(path),map_location=device,weights_only=True)
    if saved.get('status')!='trained' or saved.get('model_class')!='AccuracyRGCN':
        raise ValueError('Expected an explicitly trained AccuracyRGCN checkpoint')
    config=saved['config']
    model=AccuracyRGCN(saved['schema'],saved['tasks'],saved['target_calibration'],**{k:config[k] for k in KEYS}).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    return model,saved

@torch.no_grad()
def predict_physical(model,batch):
    """Return physical-unit predictions, keeping the supplied original mask."""
    device=next(model.parameters()).device;batch=batch.to(device);prediction=model(batch)
    physical=inverse_symlog(prediction)*batch.scales.to(torch.float64)
    if not torch.isfinite(physical[batch.mask]).all():raise ValueError('Nonfinite observed physical prediction')
    return {'prediction':physical,'mask':batch.mask,'sample_ids':batch.sample_ids,'semantic_ids':batch.semantic_ids}
