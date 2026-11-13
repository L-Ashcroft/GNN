"""Independent bounded-memory accuracy revision; immutable old experiments."""
from pathlib import Path
from datetime import datetime, timezone
import os
import hashlib
import json
import time
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/rgcn_accuracy_v3'
CONTROL = ROOT / 'monitoring/rgcn_accuracy_v3'
RAW = ROOT / 'data/model_ready/fusion_v2_training_v1'
META = ROOT / 'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def atomic(p, value):
    p = Path(p); assert p.resolve().is_relative_to(ROOT)
    p.parent.mkdir(parents=True, exist_ok=True); q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for _ in range(100):
        try: os.replace(q, p); return
        except PermissionError: time.sleep(.1)
    raise RuntimeError('Atomic publication failed')
def freeze(p, value):
    if Path(p).exists(): assert read(p) == value, 'Changed recipe; use a new output version'
    else: atomic(p, value)
def save_npz(p, state):
    p = Path(p); q = p.with_suffix('.npz.tmp')
    with q.open('wb') as f:
        import numpy as np
        np.savez_compressed(f, **state)
    os.replace(q, p)
