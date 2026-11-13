"""Check committed release manifests and final data; never fit on or infer on test data."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--all', action='store_true', help='Include optional upstream evidence')
    args = parser.parse_args()
    config = json.loads((ROOT / 'distribution/manifest.json').read_text(encoding='utf-8'))
    count = 0
    for archive in config['archives']:
        if archive['id'] == 'upstream-evidence' and not args.all:
            continue
        files = json.loads((ROOT / archive['file_manifest']).read_text(encoding='utf-8'))
        for item in files:
            p = ROOT / item['path']
            if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise SystemExit('Missing or changed release member: ' + item['path'])
            count += 1
    for name, expected in json.loads((ROOT / 'reproducibility/source_fingerprints.json').read_text(encoding='utf-8')).items():
        if sha(ROOT / name) != expected:
            raise SystemExit('Source fingerprint changed: ' + name)
    print('Release files and source fingerprints verified:', count, flush=True)
    if args.data:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pa.set_cpu_count(1)
        sys.path.insert(0, str(ROOT))
        from src.dataset_fusion import FusionDataset
        public = FusionDataset()
        seen = set(); sources = Counter(); tops = Counter()
        for row in public.samples(materialize=False):
            assert row['sample_id'] not in seen
            seen.add(row['sample_id']); sources[row['source_dataset']] += 1; tops[row['topology_id']] += 1
            assert len(row['performance']) == len(row['mask']) == len(row['semantic_ids']) == 16
            assert all((y is not None and math.isfinite(y)) if m else y is None
                       for y, m in zip(row['performance'], row['mask']))
        assert len(seen) == 500000 and dict(sources) == {'AICircuit':100000,'FALCON':200000,'LOADBench':200000}
        assert len(tops) == 121
        del seen
        for source in sources:
            graph_row = next(public.samples(source=source, materialize=True))
            assert graph_row['edge_types'] and graph_row['graph']['devices']
        manifest = json.loads((ROOT / 'data/generated/analoggym_personal_20k_v1/dataset.json').read_text(encoding='utf-8'))
        ids = set(); counts = Counter()
        for shard in manifest['shards']:
            rows = 0
            for b in pq.ParquetFile(ROOT / shard['path']).iter_batches(batch_size=128, use_threads=False):
                for row in b.to_pylist():
                    assert row['sample_id'] not in ids
                    ids.add(row['sample_id']); counts[row['topology_id']] += 1; rows += 1
                    assert len(row['performance']) == len(row['mask']) == len(row['semantic_ids']) <= 16
                    assert all((y is not None and math.isfinite(y)) if m else y is None
                               for y, m in zip(row['performance'], row['mask']))
                    assert (ROOT / row['graph']).exists()
            assert rows == shard['rows']
        assert len(ids) == 20000 and len(counts) == 10 and set(counts.values()) == {2000}
        print(json.dumps({'public_rows':500000,'public_sources':dict(sources),'public_topology_entries':len(tops),
                          'personal_rows':len(ids),'personal_topologies':len(counts),'raw_simulation_archives_checked':False,
                          'new_test_predictions_generated':False}), flush=True)


if __name__ == '__main__':
    main()
