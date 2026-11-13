"""Validation-selected physical-output adapter; preserves every raw label/mask."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BoundedPhysicalReadout:
    """Apply AFTER the existing corrected model's raw-unit predictions.

    This adapter is for known-topology ID only. It never creates a new target
    head, changes an observed mask, or admits a new cross-source contract.
    """
    def __init__(self, publication=None):
        path = Path(publication) if publication else ROOT / 'artifacts/research_repair_v1/loadbench_shared_v2/deployment.json'
        self.publication = read(path)
        for rel, expected in self.publication['sources'].items():
            if sha(ROOT / rel) != expected:
                raise ValueError('Pinned readout changed: ' + rel)
        tasks = read(ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json')
        self.groups = set(tasks); self.bounds = {}
        for group, item in self.publication['groups'].items():
            with np.load(ROOT / item['bounds'], allow_pickle=False) as state:
                low, high = state[group + '__own_low'], state[group + '__own_high']
                observed = state[group + '__observed'].copy()
            scales = np.array([tasks[group]['slots'].get(str(j), {}).get('scale', 1.) for j in range(16)])
            lo = np.sign(low) * np.expm1(abs(low)) * scales
            hi = np.sign(high) * np.expm1(abs(high)) * scales
            self.bounds[group] = lo, hi, observed

    def __call__(self, supervision_group, prediction_physical):
        if supervision_group not in self.groups:
            raise ValueError('Untrained supervision contract; no zero-shot fallback')
        value = np.asarray(prediction_physical)
        if value.ndim != 2 or value.shape[1] != 16 or not np.isfinite(value).all():
            raise ValueError('Expected finite [N,16] RAW PHYSICAL UNIT predictions')
        result = value.copy()
        if supervision_group in self.bounds:
            lo, hi, observed = self.bounds[supervision_group]
            result[:, observed] = np.clip(result[:, observed], lo[observed], hi[observed])
        return result
