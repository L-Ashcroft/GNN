"""Independent repair run: immutable sources, bounded memory, atomic records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/research_repair_v1'
CONTROL = ROOT / 'monitoring/research_repair_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.repair.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for _ in range(100):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError('Atomic publication failed: ' + str(path))


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, 'Immutable recipe changed; create a new version'
    else:
        atomic(path, value)


def gate(gib=1.6):
    import psutil
    if psutil.virtual_memory().available < gib * 2**30:
        raise SystemExit(75)


def metrics(y, p, mask, rules, topology):
    import numpy as np
    rows = []
    log_units = {'dB', 'dBm', 'dBc/Hz', 'source_label_dBm_compression_function'}
    for label, rule in rules.items():
        j = int(label); observed = mask[:, j]
        a, b = y[observed, j], p[observed, j]
        if not len(a):
            continue
        assert np.isfinite(a).all() and np.isfinite(b).all()
        err = b - a; sst = np.square(a - a.mean()).sum()
        relative = (np.abs(a) > 1e-8 * rule['scale']) & (rule['unit'] not in log_units)
        rr = np.abs(err[relative] / a[relative])
        rows.append({'topology_id': topology, 'slot': j, 'original_name': rule['original_name'],
                     'unit': rule['unit'], 'count': len(a), 'relative_count': int(relative.sum()),
                     'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt(np.square(err).mean())),
                     'r2': float(1 - np.square(err).sum() / sst) if len(a) > 1 and sst > 0 else None,
                     'hit5': float((rr <= .05).mean()) if len(rr) else None,
                     'hit10': float((rr <= .1).mean()) if len(rr) else None,
                     'hit_abs_1_unit': float((np.abs(err) <= 1).mean()) if rule['unit'] in log_units else None,
                     'target_std': float(a.std()), 'prediction_std': float(b.std())})
    return rows


def summarize(rows):
    import numpy as np
    r = [x['r2'] for x in rows if x['r2'] is not None]
    h = [x['hit5'] for x in rows if x['hit5'] is not None]
    return {'macro_hit5': float(np.mean(h)) if h else None,
            'median_r2': float(np.median(r)) if r else None,
            'macro_r2': float(np.mean(r)) if r else None,
            'negative_r2_targets': sum(x < 0 for x in r), 'target_pairs': len(rows)}
