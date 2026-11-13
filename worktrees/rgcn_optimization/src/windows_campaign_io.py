"""Atomic campaign writes tolerant of transient Windows sharing violations."""
import json
import os
from pathlib import Path
import tempfile
import time
from .analoggym_campaign import inside


def replace_with_retry(source,target,timeout=15.):
    """Retry only Win32 access/sharing/lock errors; preserve source on failure."""
    source=inside(source);target=inside(target)
    deadline=time.monotonic()+timeout;delay=.05
    while True:
        try:
            os.replace(source,target)
            return
        except OSError as error:
            if getattr(error,'winerror',None) not in {5,32,33} or time.monotonic()>=deadline:
                raise
            time.sleep(min(delay,max(0.,deadline-time.monotonic())))
            delay=min(.5,delay*2)


def atomic_json(path,obj):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    # Unique names prevent concurrent status/report writers from sharing a temp file.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
            prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    replace_with_retry(temporary,path)
