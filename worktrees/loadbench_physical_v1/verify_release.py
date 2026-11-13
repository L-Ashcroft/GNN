"""Independent identity/coverage/mask/source and recovery checks for audit outputs."""
from pathlib import Path
import hashlib,json
import pyarrow.parquet as pq
import psutil
from load import QualityView

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'data/audit/loadbench_physical_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m=read(BASE/'manifest.json');seen=set();review=0;rows=0
    for batch in pq.ParquetFile(BASE/'quality_index.parquet').iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            assert r['sample_id'] not in seen;seen.add(r['sample_id']);rows+=1
            assert r['needs_review']==bool(r['review_flags']);review+=int(r['needs_review'])
    assert rows==500000 and review==34788
    for batch in pq.ParquetFile(ROOT/'data/fusion_v2/index.parquet').iter_batches(batch_size=2048,columns=['sample_id']):
        for sid in batch.column(0).to_pylist():assert sid in seen;seen.remove(sid)
    assert not seen
    for file,digest in m['source_partition_sha256'].items():assert sha(ROOT/file)==digest
    for file,digest in m['output_sha256'].items():assert sha(ROOT/file)==digest
    assert pq.ParquetFile(BASE/'review_references.parquet').metadata.num_rows==34788
    assert pq.ParquetFile(BASE/'no_trigger_references.parquet').metadata.num_rows==465212
    v=QualityView();examples=0
    try:
        next(v.references(role='test'));raise AssertionError('Test protection absent')
    except ValueError:pass
    for source in ['FALCON','AICircuit','LOADBench']:
        for row in v.samples(role='train',source=source,limit=2):
            assert len(row['performance'])==len(row['mask'])==16
            for value,mask in zip(row['performance'],row['mask']):assert bool(mask)==(value is not None)
            examples+=1
        v.close()
    assert examples==6
    replay=read(BASE/'native_wrapper_units_v2_summary.json')
    assert len(replay['results'])==4 and all(r['state']=='simulated' for r in replay['results'])
    abnormal=next(r for r in replay['results'] if r['label']=='extreme_130')['analysis']
    assert abnormal['gain_max_dB']<0 and abnormal['no_measured_unity_crossing']
    assert abnormal['official_function_on_replayed_waveform']['ft']>abnormal['frequency_max_Hz']
    # Preserve original-phase branch as evidence, not a certified corrected label.
    controls=[]
    for r in replay['results']:
        if r['label']=='extreme_130':continue
        a=r['analysis'];official=a['official_function_on_replayed_waveform'];archive=a['archived']
        controls.append({'label':r['label'],'gain_absolute_difference_dB':abs(official['dc_gain']-archive['dc_gain']),
                         'ft_relative_difference':abs(official['ft']/archive['ft']-1),
                         'phase_absolute_difference_degree':abs(official['phasemargin']-archive['phasemargin'])})
    pause=read(BASE/'pause_record.json');assert sha(ROOT/'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/recipe.json')==pause['recipe_sha256']
    s=read(ROOT/'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/status.json');p=psutil.Process(s['pid'])
    assert abs(p.create_time()-s['create_time'])<.01 and s['state'] in {'training','validating','epoch_complete'}
    assert s['epoch']>=pause['saved_status']['epoch'] and s['updates']>=pause['saved_status']['updates']
    result={'passed':True,'unique_original_IDs':rows,'review_rows':review,'no_trigger_rows':rows-review,
            'sample_interface_examples':examples,'Mask_or_label_files_modified':False,'native_replays':4,
            'normal_control_differences':controls,'original_training_resumed_same_recipe':True,
            'training_status_at_check':{k:s.get(k) for k in ['state','epoch','batch','best_epoch','best_compatible_validation_hit5']}}
    (BASE/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
