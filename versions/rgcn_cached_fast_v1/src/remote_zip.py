"""Read ZIP/ZIP64 members via verified public HTTP byte ranges, never whole ZIPs."""
import io
import re
import time
import urllib.request


class HTTPRangeReader(io.RawIOBase):
    def __init__(self,url,size,block_size=65536):
        self.url=url; self.size=size; self.position=0; self.block_size=block_size
        self.cache={}; self.bytes_downloaded=0; self.requests=0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative range seek')
        return self.position

    def fetch(self,start,end):
        if end<=start:return b''
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end-1}',
            'User-Agent':'EDA-LOADBench-audit/1.0','Accept-Encoding':'identity'})
        last=None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request,timeout=90) as response:
                    status=response.status; cr=response.headers.get('Content-Range','')
                    if status!=206:raise ValueError('Server did not honor Range; whole download refused: '+str(status))
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if not match or tuple(map(int,match.groups()))!=(start,end-1,self.size):
                        raise ValueError('Incorrect byte range: '+cr)
                    data=response.read(end-start+1)
                    if len(data)!=end-start:raise ValueError('Truncated/oversized range')
                self.requests+=1;self.bytes_downloaded+=len(data)
                return data
            except (OSError,TimeoutError,ValueError) as error:
                last=error;time.sleep(min(2**attempt,8))
        raise last

    def read(self,length=-1):
        if length<0:length=self.size-self.position
        end=min(self.size,self.position+length)
        if end<=self.position:return b''
        # Metadata reads benefit from small blocks. Large member reads use exact
        # ranges; no accidental fetch of an entire multi-GB archive is permitted.
        if end-self.position>2*self.block_size:
            data=self.fetch(self.position,end);self.position=end;return data
        chunks=[]
        while self.position<end:
            block=self.position//self.block_size;start=block*self.block_size
            if block not in self.cache:
                self.cache[block]=self.fetch(start,min(self.size,start+self.block_size))
                if len(self.cache)>64:
                    first=next(iter(self.cache))
                    if first!=block:del self.cache[first]
            off=self.position-start; n=min(end-self.position,len(self.cache[block])-off)
            chunks.append(self.cache[block][off:off+n]);self.position+=n
        return b''.join(chunks)
