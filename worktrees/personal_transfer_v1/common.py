"""Isolated personal-domain experiments; native Windows and immutable inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VERSION = ROOT / 'versions/rgcn_calibrated_head_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VERSION))
sys.path.insert(0, str(HERE))
import src.common as source_common
source_common.ROOT = ROOT

CACHE = ROOT / 'data/model_ready/personal_transfer_v1'
OUTPUT = ROOT / 'artifacts/experiment4_personal_transfer_v1'
SOURCE = ROOT / 'data/generated/analoggym_personal_20k_v1'
META = ROOT / 'data/model_ready/analoggym_personal_20k_v1'
PRETRAINED = ROOT / 'artifacts/experiment1_autonomous_head_v1/best.pt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def replace(temp, target):
    for _ in range(80):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError('Atomic replacement failed: ' + str(target))


def atomic(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace(temp, path)


def pins():
    files = sorted(HERE.glob('*.py')) + sorted((VERSION / 'src').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
