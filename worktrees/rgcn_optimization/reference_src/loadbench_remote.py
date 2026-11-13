"""CRC-verified selective access to LOADBench nested archives.

The published outer ZIP wraps already-compressed all_sims.zip files using
byte-aligned DEFLATE stored blocks. We validate this encoding and reconstruct
only small windows around the inner central directory and chosen members.
We never assume arbitrary DEFLATE streams are seekable.
"""
import binascii
import io
import struct
import zipfile
import zlib


def payload_offset(remote,member):
    offset=member['header_offset'];header=remote.fetch(offset,offset+30)
    if header[:4]!=b'PK\x03\x04':raise ValueError('Bad ZIP local header')
    n,e=struct.unpack_from('<HH',header,26)
    return offset+30+n+e


def read_member(remote,member):
    start=payload_offset(remote,member)
    raw=remote.fetch(start,start+member['compressed_size'])
    data=zlib.decompress(raw,-15) if member['method']==8 else raw
    if len(data)!=member['size'] or binascii.crc32(data)&0xffffffff!=member['crc32']:
        raise ValueError('ZIP member CRC/size mismatch: '+member['name'])
    return data


def decode_stored_window(data,require_end=False):
    """Locate three consistent stored-block boundaries, then strip all headers."""
    for start in range(min(65540,max(0,len(data)-10))):
        cursor=start; pieces=[];blocks=0;ended=False
        while cursor+5<=len(data):
            flag=data[cursor]
            if flag not in (0,1):break
            n,inverse=struct.unpack_from('<HH',data,cursor+1)
            if n^inverse!=65535:break
            if cursor+5+n>len(data):
                if not require_end and blocks>=3:
                    pieces.append(data[cursor+5:]);return b''.join(pieces)
                break
            pieces.append(data[cursor+5:cursor+5+n]);blocks+=1;cursor+=5+n
            if flag==1:
                ended=True;break
        if blocks>=3 and (ended if require_end else ended or cursor>=len(data)-4):
            return b''.join(pieces)
    raise ValueError('Unsupported/non-stored outer DEFLATE window; refuse guessed offsets')


class NestedStoredZip:
    def __init__(self,remote,outer_member):
        self.remote=remote;self.outer=outer_member;self.start=payload_offset(remote,outer_member)
        if outer_member['method']!=8 or outer_member['compressed_size']<outer_member['size']:
            raise ValueError('Expected outer stored-block DEFLATE wrapper')

    def directory(self):
        size=self.outer['compressed_size']; n=min(size,524288)
        data=self.remote.fetch(self.start+size-n,self.start+size)
        tail=decode_stored_window(data,require_end=True)
        shift=self.outer['size']-len(tail)
        with zipfile.ZipFile(io.BytesIO(tail)) as z:
            return [{'name':m.filename,'size':m.file_size,'compressed_size':m.compress_size,
                     'header_offset':m.header_offset+shift,'crc32':m.CRC,'method':m.compress_type,
                     'flag_bits':m.flag_bits} for m in z.infolist()]

    def member(self,m):
        # For a stored-block wrapper, global compressed/uncompressed offsets
        # differ only by 5 bytes per block. A guard window tolerates uneven
        # block sizes; the member's full local header and CRC are authoritative.
        ratio=self.outer['compressed_size']/self.outer['size']
        guess=int(m['header_offset']*ratio)
        guard=131072
        lo=max(0,guess-guard)
        hi=min(self.outer['compressed_size'],guess+guard+int(m['compressed_size']*ratio)+65536)
        window=self.remote.fetch(self.start+lo,self.start+hi)
        decoded=decode_stored_window(window)
        name=m['name'].encode('utf-8'); candidates=[];cursor=0
        while True:
            p=decoded.find(b'PK\x03\x04',cursor)
            if p<0:break
            cursor=p+1
            if p+30>len(decoded):continue
            n,e=struct.unpack_from('<HH',decoded,p+26)
            if decoded[p+30:p+30+n]==name:candidates.append(p+30+n+e)
        for offset in candidates:
            raw=decoded[offset:offset+m['compressed_size']]
            if len(raw)!=m['compressed_size']:continue
            try:data=zlib.decompress(raw,-15) if m['method']==8 else raw
            except zlib.error:continue
            if len(data)==m['size'] and binascii.crc32(data)&0xffffffff==m['crc32']:
                return data
        raise ValueError('Inner member not found/CRC mismatch inside verified block window: '+m['name'])
