"""Download, verify and reconstruct the immutable dataset/model release (stdlib only)."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, dest, size, digest):
    if dest.exists() and dest.stat().st_size == size and sha(dest) == digest:
        return
    temporary = dest.with_name(dest.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'GNN-research-release'})
    with urllib.request.urlopen(request, timeout=180) as src, temporary.open('wb') as out:
        shutil.copyfileobj(src, out, 1024**2)
    if temporary.stat().st_size != size or sha(temporary) != digest:
        raise ValueError('Download checksum mismatch: ' + dest.name)
    os.replace(temporary, dest)


def restore(component, config, source, download, validate_only=False):
    for part in component['parts']:
        name = part['name']
        path = source / name
        if download:
            url = f'https://github.com/{config["repository"]}/releases/download/{config["tag"]}/{name}'
            fetch(url, path, part['bytes'], part['sha256'])
        if not path.exists() or path.stat().st_size != part['bytes'] or sha(path) != part['sha256']:
            raise ValueError('Missing or invalid part: ' + name)
    archive = source / component['archive']
    with archive.open('wb') as target:
        for part in component['parts']:
            with (source / part['name']).open('rb') as original:
                shutil.copyfileobj(original, target, 1024**2)
    if archive.stat().st_size != component['bytes'] or sha(archive) != component['sha256']:
        raise ValueError('Archive checksum mismatch')
    expected = json.loads((ROOT / component['file_manifest']).read_text(encoding='utf-8'))
    records = {item['path']: item for item in expected}
    restored = 0
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(records):
            raise ValueError('Archive member list does not match committed manifest')
        # Check all conflicts before extracting anything. Never replace research results.
        for name in names:
            dest = (ROOT / name).resolve()
            if not dest.is_relative_to(ROOT.resolve()) or '.git' in Path(name).parts:
                raise ValueError('Unsafe archive member: ' + name)
            if dest.exists() and (not dest.is_file() or sha(dest) != records[name]['sha256']):
                raise FileExistsError('Existing file differs; use a clean clone: ' + name)
        for name in names:
            dest = ROOT / name
            with z.open(name) as original:
                if validate_only or dest.exists():
                    if hashlib.file_digest(original, 'sha256').hexdigest() != records[name]['sha256']:
                        raise ValueError('Member checksum mismatch: ' + name)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix='.restore-', dir=dest.parent, delete=False) as target:
                    temp = Path(target.name)
                    shutil.copyfileobj(original, target, 1024**2)
                if temp.stat().st_size != records[name]['bytes'] or sha(temp) != records[name]['sha256']:
                    temp.unlink()
                    raise ValueError('Member checksum mismatch: ' + name)
                os.replace(temp, dest)
                restored += 1
    archive.unlink()
    print(f'{component["id"]}: verified {len(names)} files; restored {restored}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--components', nargs='+')
    parser.add_argument('--all', action='store_true', help='Include optional upstream label-audit evidence')
    parser.add_argument('--from-dir', type=Path, help='Use previously downloaded release parts')
    parser.add_argument('--validate-archives', action='store_true', help='Verify members without extracting')
    args = parser.parse_args()
    config = json.loads((ROOT / 'distribution/manifest.json').read_text(encoding='utf-8'))
    available = {x['id']: x for x in config['archives']}
    if args.list:
        for key, item in available.items():
            print(f'{key}: {item["file_count"]} files, {item["bytes"]/2**20:.2f} MiB download, {len(item["parts"])} parts')
        return
    if args.components:
        chosen = args.components
    elif args.all:
        chosen = list(available)
    else:
        chosen = ['public-data', 'personal-data', 'model-inputs', 'experiment-results']
    if not set(chosen) <= set(available):
        raise SystemExit('Unknown component; use --list')
    source = args.from_dir.resolve() if args.from_dir else ROOT / '.downloads' / config['tag']
    source.mkdir(parents=True, exist_ok=True)
    for key in chosen:
        restore(available[key], config, source, args.from_dir is None, args.validate_archives)


if __name__ == '__main__':
    main()
