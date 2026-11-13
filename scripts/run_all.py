"""Sequential, fail-fast entry point after source acquisition and dependency installation."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main():
    stages = [
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['scripts/run_pipeline.py', '--limit-per-topology', '80', '--target', '1500', '--output', 'data/smoke', '--splits', 'data/smoke_splits'],
        ['scripts/check_quality.py', '--output', 'data/smoke', '--splits', 'data/smoke_splits'],
        ['scripts/run_pipeline.py'],
        ['scripts/check_quality.py'],
        ['scripts/check_source_overlap.py'],
        ['scripts/run_pipeline.py', '--from-converted', 'data/processed', '--require-complete-inputs', '--output', 'data/strict', '--splits', 'data/strict_splits'],
        ['scripts/check_quality.py', '--output', 'data/strict', '--splits', 'data/strict_splits'],
        ['scripts/check_loading.py'],
        ['scripts/verify_reproducibility.py'],
        ['scripts/build_reports.py'],
        ['scripts/acquire_sources.py', '--check-only'],
    ]
    for stage in stages:
        print('RUN', ' '.join(stage), flush=True)
        subprocess.run([sys.executable, *stage], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
