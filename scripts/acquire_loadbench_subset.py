"""Checkpointed ~small candidate pool covering every official LOADBench topology."""
import binascii,gzip,hashlib,io,json,math,struct,sys,time,zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import dump,read,rel
from src.remote_zip import HTTPRangeReader
from src.loadbench_remote import payload_offset,decode_stored_window,read_member
from src.load_loadbench import quality_mask

BASE=ROOT/'data/raw/LOADBench';DEST=BASE/'subset'
ASSETS={'CombinedTestBench.cir','params.cir','.spiceinit','standalone_tb_extract.py','partitioning.xml','struct_recognition.xml','param_match.txt','FinalPOFfull.csv'}


def topology_job(source,topology,members):
    folder=DEST/topology
    if (folder/'acquisition.json').exists():
        try:return read(folder/'acquisition.json')
        except json.JSONDecodeError:(folder/'acquisition.json').rename(folder/'acquisition.interrupted.json')
    folder.mkdir(parents=True,exist_ok=True)
    remote=HTTPRangeReader(source['links']['self'],source['size'])
    selected=[m for m in members if m['name'].split('/')[-1] in ASSETS]
    # Coalesce adjacent small members, but never bridge a large all_sims archive.
    batches=[]
    for m in sorted(selected,key=lambda x:x['header_offset']):
        lo=m['header_offset'];hi=lo+30+len(m['name'].encode())+65536+m['compressed_size']
        if batches and lo-batches[-1][1]<131072:batches[-1][1]=max(hi,batches[-1][1]);batches[-1][2].append(m)
        else:batches.append([lo,hi,[m]])
    proofs=[]
    for lo,hi,batch in batches:
        hi=min(source['size'],hi);data=remote.fetch(lo,hi)
        for m in batch:
            at=m['header_offset']-lo
            assert data[at:at+4]==b'PK\x03\x04'
            n,e=struct.unpack_from('<HH',data,at+26);start=at+30+n+e
            raw=data[start:start+m['compressed_size']]
            body=zlib.decompress(raw,-15) if m['method']==8 else raw
            assert len(body)==m['size'] and binascii.crc32(body)&0xffffffff==m['crc32']
            target=folder/m['name'].split('/')[-1];target.write_bytes(body)
            proofs.append({'member':m['name'],'crc32':m['crc32'],'sha256':hashlib.sha256(body).hexdigest(),'path':rel(target)})
    pof=pd.read_csv(folder/'FinalPOFfull.csv') if (folder/'FinalPOFfull.csv').exists() else pd.DataFrame()
    pof_valid=int(quality_mask(pof).sum()) if len(pof) else 0
    outer=next(m for m in members if m['name'].endswith('/all_sims.zip'))
    begin=payload_offset(remote,outer);raw=b'';frame=None
    prefix_counts=[]
    # Grow adaptively; never download the rest of a 250-700 MB all_sims member.
    for size in [262144,524288,1048576,2097152,4194304,8388608]:
        raw+=remote.fetch(begin+len(raw),begin+size)
        decoded=decode_stored_window(raw)
        assert decoded[:4]==b'PK\x03\x04'
        method=struct.unpack_from('<H',decoded,8)[0];n,e=struct.unpack_from('<HH',decoded,26)
        assert decoded[30:30+n]==b'all_sims.csv' and method==8
        output=zlib.decompressobj(-15).decompress(decoded[30+n+e:],128*1024*1024)
        if len(output)>=128*1024*1024:raise ValueError('Prefix decompression memory budget exceeded')
        output=output[:output.rfind(b'\n')+1]
        frame=pd.read_csv(io.BytesIO(output))
        valid=frame.loc[quality_mask(frame)]
        params=[c for c in frame if c.lower().startswith('param_')]
        unique=len(valid.drop_duplicates(params))
        prefix_counts.append({'compressed_prefix_bytes':size,'complete_rows':len(frame),'unique_valid_parameter_vectors':unique})
        if unique+pof_valid>=2300:break
    with gzip.GzipFile(filename=str(folder/'intermediate_prefix.csv.gz'),mode='wb',mtime=0) as f:f.write(output)
    (folder/'intermediate_prefix.deflate').write_bytes(raw)
    result={'topology':topology,'source_archive':source['key'],'source_url':source['links']['self'],
        'source_archive_size':source['size'],'source_archive_checksum':source['checksum'],
        'small_member_proofs':proofs,'pof_rows':len(pof),'pof_numerically_valid':pof_valid,
        'prefix_counts':prefix_counts,'prefix_rows':len(frame),'prefix_valid_rows':int(quality_mask(frame).sum()),
        'prefix_last_generation':int(frame.generation.max()),'prefix_first_generation':int(frame.generation.min()),
        'compressed_source_offset':begin,'compressed_prefix_bytes':len(raw),
        'compressed_prefix_sha256':hashlib.sha256(raw).hexdigest(),
        'prefix_csv_sha256':hashlib.sha256(output).hexdigest(),
        'prefix_integrity':'HTTPS verified byte ranges and recorded SHA256; full inner/outer ZIP CRC not verifiable for a partial stream. Small complete members CRC verified.',
        'network_bytes':remote.bytes_downloaded,'network_requests':remote.requests}
    dump(folder/'acquisition.json',result)
    print(topology,'pof',pof_valid,'prefix',unique,'download_MiB',round(remote.bytes_downloaded/2**20,2),flush=True)
    return result


if __name__=='__main__':
    tasks=[]
    for archive in ['OneStageOpAmps.zip','TwoStageOpAmps.zip','ThreeStageOpAmps.zip','SymmetricalOpAmps.zip']:
        directory=read(BASE/(archive+'.directory.json'))
        names=sorted({m['name'].split('/')[0] for m in directory['members'] if m['name'].endswith('/all_sims.zip')})
        for name in names:tasks.append((directory['source'],name,[m for m in directory['members'] if m['name'].startswith(name+'/')]))
    assert len(tasks)==101
    dump(BASE/'download_plan.json',{'topologies':101,'target_LOADBench_rows':200000,'candidate_target_per_topology':2300,
        'maximum_compressed_prefix_bytes_per_topology':8388608,'expected_download_MiB':[150,850],
        'sampling_population':'Complete FinalPOFfull plus explicitly recorded initial generation prefix; not uniform from entire optimization history',
        'tasks':[{'archive':s['key'],'topology':n} for s,n,m in tasks]})
    started=time.perf_counter();results=[]
    errors=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs=[pool.submit(topology_job,*job) for job in tasks]
        for job in as_completed(jobs):
            try:results.append(job.result())
            except Exception as error:
                errors.append({'type':type(error).__name__,'message':str(error)})
                dump(BASE/'download_errors.json',errors)
                print('RETRY_REQUIRED',type(error).__name__,str(error),flush=True)
            dump(BASE/'download_progress.json',{'completed':len(results),'planned':101,'errors':len(errors),'network_bytes':sum(r['network_bytes'] for r in results)})
    dump(BASE/'acquisition_summary.json',{'topologies':len(results),'network_bytes':sum(r['network_bytes'] for r in results),
        'seconds':time.perf_counter()-started,'errors':errors,'results':sorted(results,key=lambda r:r['topology'])})
    if errors:raise SystemExit('Some topologies need a resumed acquisition; completed checkpoints retained')
