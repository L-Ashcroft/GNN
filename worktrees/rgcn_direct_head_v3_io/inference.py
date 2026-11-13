"""Load a complete frozen edge-RGCN with the new direct residual heads."""
from pathlib import Path
import sys,hashlib
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load_model(checkpoint=None,device='cpu'):
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
    from model import DirectResidualHead
    import src.common as common
    common.ROOT=ROOT
    from src.model.accuracy_network import AccuracyRGCN
    path=Path(checkpoint) if checkpoint else ROOT/'artifacts/experiment1_rgcn_direct_head_v3_io/best.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['head_type']!='independent_direct_residual_v3':raise ValueError('Unknown head checkpoint format')
    parent_path=ROOT/saved['graph_checkpoint']
    if sha(parent_path)!=saved['graph_checkpoint_sha256']:raise ValueError('Parent graph checkpoint changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    keys=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']
    net=AccuracyRGCN(parent['schema'],parent['tasks'],parent['target_calibration'],**{k:parent['config'][k] for k in keys})
    net.load_state_dict(parent['model'],strict=True)
    net.head=DirectResidualHead.from_state(saved['tasks'],saved['head']);net.to(device);net.eval();return net,saved

def predict_physical(model,batch):
    with torch.no_grad():
        z=model(batch);return z.sign()*torch.expm1(z.abs())*batch.scales.to(torch.float64)
