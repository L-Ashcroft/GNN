"""Verify pinned, unmodified inputs before supplemental audits."""
import subprocess
from .common import ROOT, read


def verify_source(name):
    lock = read(ROOT / 'config/analoggym_audit_sources.lock.json')[name]
    repo = ROOT / 'third_party' / name
    base = ['git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo)]
    head = subprocess.check_output([*base, 'rev-parse', 'HEAD'], text=True).strip()
    if head != lock['commit']:
        raise RuntimeError(f'{name}: unexpected commit {head}')
    status = subprocess.check_output([*base, 'status', '--porcelain'], text=True)
    if status.strip():
        raise RuntimeError(f'{name}: modified upstream worktree; audit aborted')
    return head
