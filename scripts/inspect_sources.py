"""Read-only inventory; safe to run before installing project dependencies."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo), *args], text=True).strip()


def main():
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    result = {'environment': {'os': platform.platform(), 'python': sys.version,
              'git': subprocess.check_output(['git', '--version'], text=True).strip(),
              'disk': dict(zip(['total', 'used', 'free'], shutil.disk_usage(ROOT)))}, 'repositories': {}, 'tables': [], 'archives': []}
    for name in ['FALCON', 'AnalogGym', 'AICircuit']:
        repo = ROOT / 'third_party' / name
        paths = git(repo, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines()
        result['repositories'][name] = {'commit': git(repo, 'rev-parse', 'HEAD'), 'url': git(repo, 'remote', 'get-url', 'origin'),
            'tracked_files': len(paths), 'missing_worktree_files': [p for p in paths if not (repo / p).is_file()],
            'worktree_bytes': sum((repo / p).stat().st_size for p in paths if (repo / p).is_file()),
            'extension_counts': dict(collections.Counter(Path(p).suffix or '<none>' for p in paths))}
        for rel in paths:
            p = repo / rel
            if not p.is_file():
                continue
            if p.suffix == '.zip':
                with zipfile.ZipFile(p) as z:
                    result['archives'].append({'source': name, 'path': rel, 'bytes': p.stat().st_size,
                        'expanded_bytes': sum(i.file_size for i in z.infolist()), 'members': z.namelist()[:40], 'count': len(z.infolist())})
            if p.suffix.lower() != '.csv':
                continue
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                n = bad = 0
                missing = collections.Counter()
                rows_seen = set()
                duplicates = 0
                for row in reader:
                    n += 1
                    broken = False
                    for key, value in row.items():
                        if value is None or not value.strip():
                            missing[key] += 1
                        else:
                            try:
                                if not math.isfinite(float(value)):
                                    broken = True
                            except ValueError:
                                broken = True
                    bad += broken
                    signature = tuple(row.items())
                    duplicates += signature in rows_seen
                    rows_seen.add(signature)
            result['tables'].append({'source': name, 'path': rel, 'rows': n, 'columns': columns,
                'nonfinite_or_nonnumeric_rows': bad, 'missing_cells': dict(missing), 'exact_duplicate_rows': duplicates,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size})
    (out / 'source_inventory.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'repositories': result['repositories'], 'tables': result['tables'], 'archives': result['archives']}, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
