"""One post-selection fixed-test evaluation of the frozen local GNN model.

Uses raw graphs (not a test feature cache). Test scores never enter fitting.
Per-topology shards allow recovery without changing this frozen recipe.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,time,hashlib,psutil,msvcrt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1.25*2**30:raise SystemExit(75)
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path.insert(0,str(ROOT/'versions/rgcn_local_id_v1'));sys.path.insert(0,str(ROOT/'worktrees/autonomous_training_v1'))
from model import load_model,sha
from stream_graphs import StreamGraphs
OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1/fixed_test';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def replace(a,b):
    for i in range(100):
        try:os.replace(a,b);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Atomic replace failed: '+str(b))
def atomic(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');replace(q,p)
publication=read(ROOT/'versions/rgcn_local_id_v1/publication.json');bundle_path=ROOT/publication['bundle']
assert sha(bundle_path)==publication['bundle_sha256'] and sha(ROOT/publication['source'])==publication['source_sha256']
lock=(OUT/'evaluation.lock').open('a+b');lock.seek(0)
if not lock.read(1):lock.write(b'0');lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
state={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),'stage':'frozen_fixed_test_evaluation',
    'train_or_model_updates':False,'test_set_reused_from_previous_experiments':True,'selection_was_validation_only':True,'bundle_sha256':publication['bundle_sha256']}
def update(**v):state.update(v,updated_at=now());atomic(OUT/'status.json',state)
reader=None;start=time.perf_counter();peak=0
try:
    if (OUT/'summary.json').exists():print('Fixed test already evaluated; immutable summary retained.');raise SystemExit(0)
    update();net,bundle=load_model(bundle_path);reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1','data/model_ready/fusion_v2_frozen_e19_bf0d71ad','versions/rgcn_calibrated_head_v1',capacity=2,verify=True)
    from src.train.metrics import MetricAccumulator
    all_rows=[];counts={};done=0;timings=[];seen_ids=set();train_test_disjoint=True
    schema=pa.schema([('sample_id',pa.string()),('source_dataset',pa.string()),('supervision_group',pa.string()),
        ('prediction',pa.list_(pa.float64(),16)),('target',pa.list_(pa.float32(),16)),('mask',pa.list_(pa.bool_(),16))])
    for number,(name,spec) in enumerate(reader.manifest['topologies'].items(),1):
        receipt=OUT/(name+'.json');parquet=OUT/(name+'.parquet');folder=ROOT/spec['folder']
        if receipt.exists():
            prior=read(receipt);assert prior['bundle_sha256']==publication['bundle_sha256'] and sha(parquet)==prior['prediction_sha256'];all_rows+=prior['metrics']['per_target'];done+=prior['rows'];counts[name.split('__')[0]]=counts.get(name.split('__')[0],0)+prior['rows'];continue
        role=np.load(folder/'role.npy',mmap_mode='r');ids=np.load(folder/'sample_ids.npy',mmap_mode='r');rows=np.flatnonzero(role==2)
        train_ids=set(ids[role==0].tolist());test_ids=ids[rows].tolist();assert not train_ids&set(test_ids);assert not seen_ids&set(test_ids);seen_ids.update(test_ids)
        role._mmap.close();ids._mmap.close()
        if not len(rows):continue
        metric=MetricAccumulator(reader.tasks);tmp=parquet.with_suffix('.parquet.tmp');writer=pq.ParquetWriter(tmp,schema,compression='zstd');tick=time.perf_counter()
        try:
            with torch.no_grad():
                for offset in range(0,len(rows),128):
                    if psutil.virtual_memory().available<512*2**20:raise MemoryError('Insufficient RAM for frozen CPU evaluation; completed shards preserved')
                    selected=rows[offset:offset+128];batch=reader.batch_refs([(name,int(i)) for i in selected],'test');z=net(batch);metric.update(z,batch)
                    physical=(z.sign()*torch.expm1(z.abs())*batch.scales.to(torch.float64)).numpy()
                    if not np.isfinite(physical[batch.mask.numpy()]).all():raise ValueError('Nonfinite observed frozen test prediction')
                    table=pa.Table.from_arrays([pa.array(batch.sample_ids),pa.array([name.split('__')[0]]*len(selected)),pa.array(batch.group_ids),
                        pa.FixedSizeListArray.from_arrays(pa.array(physical.reshape(-1)),16),
                        pa.FixedSizeListArray.from_arrays(pa.array(batch.targets.numpy().reshape(-1)),16),
                        pa.FixedSizeListArray.from_arrays(pa.array(batch.mask.numpy().reshape(-1)),16)],schema=schema)
                    writer.write_table(table);peak=max(peak,psutil.Process().memory_info().rss)
                    update(state='evaluating',topology=name,completed_topologies=number-1,total_topologies=len(reader.manifest['topologies']),samples_done=done+offset+len(selected),total_test_samples=49056,elapsed_seconds=time.perf_counter()-start)
        finally:writer.close()
        replace(tmp,parquet);metrics=metric.compute();receipt_data={'topology':name,'rows':len(rows),'bundle_sha256':publication['bundle_sha256'],
            'prediction_sha256':sha(parquet),'metrics':metrics,'seconds':time.perf_counter()-tick,'train_test_ids_disjoint':True}
        atomic(receipt,receipt_data);all_rows+=metrics['per_target'];done+=len(rows);counts[name.split('__')[0]]=counts.get(name.split('__')[0],0)+len(rows)
        print(now(),number,name,'test samples',done,flush=True)
    assert done==49056 and len({(r['supervision_group'],r['semantic_id']) for r in all_rows})==len(all_rows)
    import statistics
    def summarize(rows):
        def avg(key):
            values=[r[key] for r in rows if r[key] is not None];return statistics.mean(values) if values else None
        r2=[r['r2'] for r in rows if r['r2'] is not None]
        return {'macro_hit5':avg('hit5'),'macro_hit10':avg('hit10'),'macro_r2':avg('r2'),'macro_mre':avg('mre'),
            'median_r2':statistics.median(r2) if r2 else None,'negative_r2_targets':sum(v<0 for v in r2),'target_groups':len(rows)}
    metrics=summarize(all_rows);metrics['by_source']={s:summarize([r for r in all_rows if reader.tasks[r['supervision_group']]['contract']['source_dataset']==s]) for s in ['FALCON','AICircuit','LOADBench']}
    atomic(OUT/'metrics.json',dict(metrics,per_target=all_rows))
    summary={'state':'complete','completed_at':now(),'test_samples':done,'source_counts':counts,'metrics':metrics,'bundle_sha256':publication['bundle_sha256'],
        'seconds':time.perf_counter()-start,'peak_RSS_MiB':peak/2**20,'train_test_ids_disjoint':True,'model_modified_by_test':False,
        'test_protocol_note':'Same fixed holdout as prior models, not a new blind dataset; this branch was frozen before access and will not be retuned from these scores.'}
    atomic(OUT/'summary.json',summary);update(state='complete',samples_done=done,metrics=metrics,elapsed_seconds=summary['seconds']);print(json.dumps(summary,indent=2))
except SystemExit:raise
except BaseException as e:update(state='error',error_type=type(e).__name__,error=str(e));raise
finally:
    if reader is not None:reader.close()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
