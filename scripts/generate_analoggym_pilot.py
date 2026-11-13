"""Generate at most 20 measured AnalogGym NMCF pilot candidates on native Windows."""
import argparse
import platform
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analoggym_generation import generate


def main():
    if platform.system() != 'Windows': raise RuntimeError('Native Windows required')
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='data/generated/analoggym_nmcf_pilot')
    p.add_argument('--timeout', type=int, default=120, help='Timeout seconds per analysis')
    a = p.parse_args()
    print(generate(ROOT/a.output, a.samples, a.seed, a.timeout))


if __name__ == '__main__': main()
