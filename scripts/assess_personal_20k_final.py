"""Final missingness, split, budget and model-interface audit. Never trains."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import sqlite3
import sys
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json

OUT=ROOT/'data/generated/analoggym_personal_20k_v1'


def main():
    state=read(OUT/'status.json');manifest=read(OUT/'manifest.json');snapshot=read(OUT/'dataset.json')
    assert state['complete'] and state['accepted']==20000==snapshot['sample_count']
    assert all(s['accepted']==s['target']==2000 for s in state['topologies'].values())
    recipe=read(OUT/'recipe.json');revision=read(OUT/'sampling_policy_v2.json')
    for path,expected in recipe['code_sha256'].items():assert sha(ROOT/path)==expected
    for path,expected in revision['implementation_sha256'].items():assert sha(ROOT/path)==expected
    for path,expected in read(OUT/'orchestration_recipe.json')['code_sha256'].items():assert sha(ROOT/path)==expected
    assert sha(OUT/'sampling_policy_v2.json')==manifest['sampling_policy_file_sha256']
    assert sha(OUT/'quality_policy_v1.json')==manifest['quality_policy_file_sha256']
    required=['sample_id','source_dataset','topology_id','circuit_family','process','PDK','temperature',
        'supply_voltage','testbench_id','graph','conditions_json','parameter_names','parameters','parameter_units',
        'device_features','performance','mask','semantic_ids','source_path','source_sha256','candidate','design_group',
        'recipe_hash','generation_status','sampling_policy_version','sampling_policy_sha256']
    nullable={'temperature'};nulls=Counter();counts=Counter();missing_slots=Counter();feature_missing=defaultdict(Counter)
    ids=set();designs=set();base_index={};graphs={};shard_counts=Counter();parquet_hashes={}
    for shard in snapshot['shards']:
        path=ROOT/shard['path'];table=pq.ParquetFile(path)
        assert set(required)<=set(table.schema_arrow.names)
        assert table.metadata.num_rows==shard['rows']
        parquet_hashes[shard['path']]=sha(path);shard_counts[path.parent.name]+=1
        for batch in table.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                name=row['topology_id'];counts[name]+=1
                for key in required:
                    if row[key] is None:
                        nulls[key]+=1;assert key in nullable,(name,key)
                    elif isinstance(row[key],str):assert row[key].strip(),(name,key)
                assert row['sample_id'] not in ids and row['design_group'] not in designs
                ids.add(row['sample_id']);designs.add(row['design_group'])
                base_index[row['sample_id']]={k:row[k] for k in ['topology_id','circuit_family','graph','design_group','source_path','candidate']}
                graph=graphs.setdefault(name,read(row['graph']) if name not in graphs else graphs[name])
                assert len(row['performance'])==len(row['mask'])==len(row['semantic_ids'])==16
                expected=15 if row['circuit_family']=='Amplifier' else 13
                assert row['mask']==[True]*expected+[False]*(16-expected)
                for i,(value,known,semantic) in enumerate(zip(row['performance'],row['mask'],row['semantic_ids'])):
                    assert bool(known)==(value is not None)==(semantic is not None)
                    if not known:missing_slots[(row['circuit_family'],i)]+=1
                for device,features in zip(graph['devices'],row['device_features']):
                    assert len(features)==17
                    for key,value in zip(graph['feature_names'],features):
                        if value is None:feature_missing[device['model']][key]+=1
                    model=device['model'];required_positions={'nmos':[0,1,2],'pmos':[0,1,2],
                        'resistor':[3],'capacitor':[4],'inductor':[5]}.get(model,[])
                    assert all(features[i] is not None and features[i]>0 for i in required_positions),(name,model)
    assert len(ids)==len(designs)==20000 and set(counts.values())=={2000}
    index=pq.read_table(OUT/'personal_index.parquet').to_pylist();split=pq.read_table(OUT/'splits/id.parquet').to_pylist()
    assert len(index)==len(split)==20000
    assert len({r['sample_id'] for r in index})==len({r['sample_id'] for r in split})==20000
    assert {r['sample_id'] for r in index}=={r['sample_id'] for r in split}==ids
    for row in index+split:
        assert all(row[key]==value for key,value in base_index[row['sample_id']].items())
    role_sets={role:{r['sample_id'] for r in split if r['role']==role} for role in ['train','validation','test']}
    assert {k:len(v) for k,v in role_sets.items()}=={'train':16000,'validation':2000,'test':2000}
    assert not role_sets['train']&role_sets['validation'] and not role_sets['train']&role_sets['test'] and not role_sets['test']&role_sets['validation']
    split_counts=Counter((r['topology_id'],r['role']) for r in split)
    for name in counts:
        assert [split_counts[(name,role)] for role in ['train','validation','test']]==[1600,200,200]
    assert len({r['design_group'] for r in split})==20000
    global_budgets=read(OUT/'splits/few_shot_global.json');per_topology=read(OUT/'splits/few_shot_per_topology.json')
    budgets=[];previous=set()
    assert [r['budget'] for r in global_budgets['budgets']]==[0,5,10,20,50,100,200,500,1000]
    for budget in global_budgets['budgets']:
        selected=set(budget['sample_ids']);assert len(selected)==len(budget['sample_ids'])==budget['budget']
        assert previous<=selected<=role_sets['train'];previous=selected
        observed=dict(Counter(base_index[s]['topology_id'] for s in selected));assert observed==budget['topology_counts']
        budgets.append({'scope':'global','labels':len(selected),'covered_topologies':len(observed)})
    previous=set()
    assert [r['budget_per_topology'] for r in per_topology['budgets']]==[5,10,20,50,100]
    for budget in per_topology['budgets']:
        selected=set(budget['sample_ids']);assert len(selected)==len(budget['sample_ids'])==budget['total_budget']
        assert previous<=selected<=role_sets['train'];previous=selected
        observed=Counter(base_index[s]['topology_id'] for s in selected)
        assert set(observed)==set(counts) and set(observed.values())=={budget['budget_per_topology']}
        budgets.append({'scope':'per_topology','k':budget['budget_per_topology'],'labels':len(selected),'covered_topologies':10})
    db=sqlite3.connect(f'file:{(OUT/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    attempts,accepted=db.execute('select count(*),sum(accepted) from attempts').fetchone()
    assert (attempts,accepted)==(state['attempts'],20000)
    accepted_pairs={(r[0],r[1]) for r in db.execute('select topology,candidate from attempts where accepted=1')}
    assert accepted_pairs=={(v['topology_id'],v['candidate']) for v in base_index.values()}
    assert not {('Song_DACFC_Pin_3',101),('Song_DACFC_Pin_3',252)}&accepted_pairs
    quality_exclusions=db.execute("select count(*) from attempts where accepted=0 and json_extract(result_json,'$.failure') like 'QualityReview:%'").fetchone()[0]
    db.close()
    completed_check=read(OUT/'quality_checks.json');gate=read(OUT/'quality_gate_checks.json')
    assert completed_check['passed'] and completed_check['samples_checked']==20000
    assert completed_check['raw_waveforms_recomputed']==dict(counts)
    assert gate['passed'] and gate['accepted_checked']==20000 and gate['all_accepted_pass_ac_gate']
    split_role={r['sample_id']:r['role'] for r in split}
    anchor_roles=[]
    for anchor in revision['centre_provenance']:
        matches=[sid for sid,r in base_index.items() if r['topology_id']=='ldo_folded_cascode' and r['candidate']==anchor['candidate']]
        anchor_roles.append({'candidate':anchor['candidate'],'role':split_role[matches[0]] if matches else 'not_in_accepted_set'})
    graph_gaps={n:[key for key in ['node_features','node_feature_names','node_annotations','feature_vocabulary','message_passing'] if key not in g] for n,g in graphs.items()}
    result={'assessed_at_UTC':datetime.now(timezone.utc).isoformat(),'completed_at_UTC':state['updated_at'],
        'samples':20000,'topologies':10,'attempts':attempts,'quarantined':attempts-accepted,
        'quality_review_exclusions':quality_exclusions,'counts':dict(counts),'shards_per_topology':dict(shard_counts),
        'required_columns':required,'top_level_null_counts':dict(nulls),'unexpected_required_missing':0,
        'mask_missing_slots':[{'family':family,'slot':slot,'rows':n} for (family,slot),n in missing_slots.items()],
        'observed_label_scalars':20000*16-sum(missing_slots.values()),'intentional_missing_label_scalars':sum(missing_slots.values()),
        'missing_features_by_device_model':{model:dict(values) for model,values in feature_missing.items()},
        'essential_device_parameter_missing':0,'sample_ids_unique':True,'design_ids_unique':True,
        'split_sample_and_design_overlap':False,'split_counts':{k:len(v) for k,v in role_sets.items()},
        'split_counts_per_topology':{n:{role:split_counts[(n,role)] for role in role_sets} for n in counts},
        'few_shot_budgets_verified':budgets,'few_shot_only_train':True,'few_shot_nested':True,
        'folded_adaptive_anchor_roles':anchor_roles,'official_loto_file_present':(OUT/'splits/loto.parquet').exists(),
        'normalization_fitted':False,'completion_full_waveform_check_verified':True,
        'gnn_graph_adapter_gaps':graph_gaps,'direct_current_gnn_ready':not any(graph_gaps.values()),
        'no_topology_family_coverage_claim_beyond_amplifier_and_ldo':True,
        'input_artifact_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
            OUT/'dataset.json',OUT/'recipe.json',OUT/'manifest.json',OUT/'personal_index.parquet',OUT/'splits/id.parquet',
            OUT/'splits/few_shot_global.json',OUT/'splits/few_shot_per_topology.json',OUT/'splits/protocols.json',
            OUT/'quality_policy_v1.json',OUT/'quality_checks.json',OUT/'quality_gate_checks.json']},
        'parquet_shard_sha256':parquet_hashes,'production_files_modified':False,'training_started':False}
    atomic_json(ROOT/'reports/analoggym_final_delivery_audit.json',result)
    print(json.dumps({k:result[k] for k in ['samples','topologies','attempts','quarantined','observed_label_scalars',
        'intentional_missing_label_scalars','unexpected_required_missing','split_counts','quality_review_exclusions',
        'folded_adaptive_anchor_roles','direct_current_gnn_ready']},indent=2))


if __name__=='__main__':main()
