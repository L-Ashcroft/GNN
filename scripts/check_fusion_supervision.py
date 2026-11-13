"""Audit every stored target against conservative supervision contracts."""
from collections import Counter
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import dump,read
from src.dataset_fusion import FusionDataset
from src.supervision_fusion import SupervisionRouter


def main():
    dataset=FusionDataset();router=SupervisionRouter(dataset.catalog)
    counts=Counter();sources={};gain=[];rows=0
    for row in dataset.samples(materialize=False):
        description=router.describe(row);key=description['supervision_group']
        counts[key]+=1;rows+=1
        sources.setdefault(key,set()).add(row['source_dataset'])
    assert rows==dataset.manifest['sample_count']
    assert all(len(s)==1 for s in sources.values())
    mapping=read(dataset.path/'performance_mapping.json')
    assert mapping['cross_dataset_approved_pairs']==[]
    for key,spec in dataset.catalog.items():
        for metric in spec['metric_mapping']:
            if 'gain' in metric['original_name'].lower():
                gain.append({'topology_id':key,**metric})
    dump(dataset.path/'supervision_audit.json',{
        'status':'passed','rows_checked':rows,'supervision_groups':len(counts),
        'cross_source_shared_targets_approved':0,
        'policy':'Separate source/topology/definition/extractor/testbench/PVT contracts; Mask only means missing.',
        'group_counts':dict(counts),'gain_metric_evidence':gain,
        'index_sha256':dataset.manifest['index_sha256'],
        'implementation_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['src/dataset_fusion.py','src/supervision_fusion.py','scripts/check_fusion_supervision.py']}})
    print('PASS',rows,'rows;',len(counts),'conservative supervision groups; no cross-source shared targets',flush=True)


if __name__=='__main__':main()
