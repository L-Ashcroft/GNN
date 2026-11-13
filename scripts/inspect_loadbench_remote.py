"""Fetch only official small metadata and remote ZIP central directories."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.remote_zip import HTTPRangeReader


if __name__=='__main__':
    folder=ROOT/'data/raw/LOADBench';folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'record_21759759.json'
    if not metadata.exists():
        with urllib.request.urlopen('https://zenodo.org/api/records/21759759',timeout=90) as response:
            raw=response.read(4*1024*1024)
        record=json.loads(raw);assert record['id']==21759759
        metadata.write_text(json.dumps(record,indent=2),encoding='utf-8')
    record=json.loads(metadata.read_text(encoding='utf-8'))
    for f in record['files']:
        if f['key'] in {'AllCombinedTestbenches.zip','reduced_sky130_combined.zip'}:
            dest=folder/f['key']
            if not dest.exists():
                content=HTTPRangeReader(f['links']['self'],f['size']).read(f['size'])
                assert len(content)==f['size']
                assert hashlib.md5(content).hexdigest()==f['checksum'].split(':')[1]
                dest.write_bytes(content)
            assert dest.stat().st_size==f['size']
            assert hashlib.md5(dest.read_bytes()).hexdigest()==f['checksum'].split(':')[1]
            print('SMALL',f['key'],dest.stat().st_size,flush=True)
            continue
        dest=folder/(f['key']+'.directory.json')
        if dest.exists():continue
        remote=HTTPRangeReader(f['links']['self'],f['size'])
        with zipfile.ZipFile(remote) as z:
            members=[{'name':m.filename,'size':m.file_size,'compressed_size':m.compress_size,
                      'header_offset':m.header_offset,'crc32':m.CRC,'method':m.compress_type,
                      'flag_bits':m.flag_bits} for m in z.infolist()]
        dest.write_text(json.dumps({'source':f,'bytes_downloaded':remote.bytes_downloaded,
            'requests':remote.requests,'members':members},indent=2),encoding='utf-8')
        print('DIRECTORY',f['key'],'members',len(members),'downloaded',remote.bytes_downloaded,
              'first_members',[(m['name'],m['size']) for m in members[:8]],flush=True)
