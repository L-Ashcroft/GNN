"""Prepare/run/status/stop a standalone 100k accepted-design simulation campaign."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import DEFAULT_OUTPUT, inside, read
from src.analoggym_campaign_runner import initialize, run


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','run','status','stop'])
    p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--qualification',default='data/generated/analoggym_standard_v4')
    p.add_argument('--target',type=int,default=100000); p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); output=inside(ROOT/a.output)
    if a.command=='prepare':
        if a.target<1 or not 1<=a.workers<=8: raise ValueError('Invalid resource/count arguments')
        print(initialize(output,ROOT/a.qualification,a.target,a.workers))
    elif a.command=='run': run(output)
    elif a.command=='stop':
        (output/'STOP').write_text('Finish in-flight simulations and exit; delete this file to resume.\n',encoding='utf-8')
        print('Stop requested; waiting simulations will finish and checkpoint.')
    else: print(json.dumps(read(output/'status.json'),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
