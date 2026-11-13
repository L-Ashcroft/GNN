"""Freeze the validation-selected ID model before any test evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
selection=read(OUT/'selection.json');checks=read(OUT/'checks.json');assert selection['selected']['passed'] and checks['raw_graph_cached_readout_tolerance_passed']
assert checks['training_index_no_validation_overlap'] and checks['training_neighbor_rows_verified']==400083
assert sha(HERE/'model.py')==selection['policy']['model_sha256']
target=ROOT/'versions/rgcn_local_id_v1';target.mkdir(parents=True,exist_ok=True)
if (target/'model.py').exists():assert sha(target/'model.py')==sha(HERE/'model.py')
else:shutil.copy2(HERE/'model.py',target/'model.py')
publication={'published_at':datetime.now(timezone.utc).isoformat(),'scope':'Known-contract public fusion ID experiment only',
    'model_variant':'Frozen4layer48relation RGCN + canonical-device/embedding local8 decoder;5small-contract groups use old neural head',
    'bundle':'artifacts/experiment1_rgcn_local_head_v1/bundle.json','bundle_sha256':sha(OUT/'bundle.json'),
    'source':'versions/rgcn_local_id_v1/model.py','source_sha256':sha(target/'model.py'),
    'validation':selection['selected']['validation'],'validation_only_selection':True,'test_evaluated_before_publication':False,
    'index_no_validation_overlap':True,'inference_checks_sha256':sha(OUT/'checks.json'),
    'previous_models_preserved':['artifacts/experiment1_autonomous_head_v1/best.pt','artifacts/experiment1_rgcn_direct_head_v3_io/best.pt'],
    'limitations':['Unknown contracts rejected','Canonical net/device ordering required','No zero-shot LOTO claim','One seed and one ID split; not an engineering pass guarantee']}
if (target/'publication.json').exists():
    prior=read(target/'publication.json');assert prior['bundle_sha256']==publication['bundle_sha256'] and prior['source_sha256']==publication['source_sha256']
else:(target/'publication.json').write_text(json.dumps(publication,indent=2),encoding='utf-8')
print('Published immutable validation-selected ID model. Test has not been evaluated.')
