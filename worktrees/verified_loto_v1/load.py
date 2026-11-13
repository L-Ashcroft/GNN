"""Load a prepared fold's sample metadata and canonical graph references."""
from pathlib import Path
import json,hashlib
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load_fold(fold_id,role='train'):
    if role not in {'train','validation','test','unused'}:raise ValueError('Unknown split role')
    manifest=json.loads((ROOT/'data/fusion_v2/splits/verified_loto_v1/manifest.json').read_text())
    fold=next(f for f in manifest['folds'] if f['fold_id']==fold_id);path=ROOT/fold['path']
    if sha(path)!=fold['sha256'] or sha(ROOT/manifest['source_split'])!=manifest['source_split_sha256']:raise ValueError('Fold/source split changed')
    with np.load(path,allow_pickle=False) as index:
        at=index[role+'_positions'];references=index[role+'_references'] if role!='unused' else None
    table=pq.read_table(ROOT/manifest['source_split'],columns=['sample_id','design_group','topology_id','connectivity_group']).take(at)
    if role in {'train','validation'} and fold['heldout_structure'] in table['connectivity_group'].to_pylist():raise ValueError('Heldout circuit entered training/validation')
    refs=None if references is None else [(manifest['topologies'][int(a)],int(b)) for a,b in references]
    return table,refs,fold
