"""Pinned, resumable Git acquisition on native Windows; no global Git config writes."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def run(repo, *args, **kwargs):
    return subprocess.run(['git', '-c', 'credential.helper=', '-c', 'core.longpaths=true',
                           '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo), *args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--lock', default='config/sources.lock.json')
    args = parser.parse_args()
    sources = json.loads((ROOT / args.lock).read_text())
    for name, info in sources.items():
        repo = ROOT / 'third_party' / name
        fresh = not (repo / '.git').exists()
        if not (repo / '.git').exists():
            if args.check_only:
                raise FileNotFoundError(repo)
            repo.parent.mkdir(exist_ok=True)
            subprocess.run(['git', '-c', 'credential.helper=', '-c', 'core.longpaths=true', 'clone',
                            '--depth', '1', '--no-checkout', info['url'], str(repo)], check=True)
        current = run(repo, 'rev-parse', 'HEAD', capture_output=True, text=True).stdout.strip()
        if not fresh:
            status = run(repo, 'status', '--porcelain', capture_output=True, text=True).stdout
            if status.strip():
                raise RuntimeError('Existing upstream worktree modified; refusing to alter its index: ' + name)
        if current != info['commit']:
            if args.check_only or not fresh:
                raise RuntimeError('Commit mismatch: ' + name)
            run(repo, 'fetch', '--depth', '1', 'origin', info['commit'])
            # Fresh --no-checkout clone: move only the local reference before index/export.
            run(repo, 'update-ref', 'HEAD', info['commit'])
        names = run(repo, 'ls-tree', '-r', '--name-only', 'HEAD', capture_output=True, text=True).stdout.splitlines()
        bad = [p for p in names if any(x.endswith(('.', ' ')) for x in p.split('/'))]
        if not args.check_only:
            # Disable NTFS validation ONLY while populating the index, not while writing files.
            run(repo, '-c', 'core.protectNTFS=false', 'read-tree', 'HEAD')
            if bad:
                run(repo, '-c', 'core.protectNTFS=false', 'update-index', '--skip-worktree', '-z', '--stdin',
                    input=b'\0'.join(p.encode() for p in bad) + b'\0')
            # checkout-index respects skip-worktree; prohibited Windows filenames never materialize.
            missing_valid = [p for p in names if p not in bad and not (repo / p).is_file()]
            if missing_valid:
                run(repo, 'checkout-index', '-z', '--stdin', input=b'\0'.join(p.encode() for p in missing_valid) + b'\0')
        status = run(repo, 'status', '--porcelain', capture_output=True, text=True).stdout
        if status.strip():
            raise RuntimeError('Upstream worktree is modified; refusing to overwrite: ' + name + '\n' + status)
        missing = [p for p in names if not (repo / p).is_file()]
        if missing != bad:
            raise RuntimeError('Unexpected missing upstream files: ' + name)
        print(name, info['commit'], 'clean;', len(bad), 'Windows-only filename omissions')


if __name__ == '__main__':
    main()
