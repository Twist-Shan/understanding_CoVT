"""Colab entrypoint for fixed, zero and shuffled-state decoder controls."""
import argparse
import json
import os
from pathlib import Path

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run controls on Colab only.')

from covt_pilot.controls import run_controls

parser=argparse.ArgumentParser()
parser.add_argument('--e0-dir',required=True)
parser.add_argument('--decoder-dir',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--device',default='cuda')
parser.add_argument('--repeats',type=int,default=5)
print(json.dumps(run_controls(**vars(parser.parse_args())),indent=2))
