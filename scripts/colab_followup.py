"""Colab FP32 E0 plus terminal-state controls, fresh fixed-seed scene families."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Run the follow-up on Colab only.')

from huggingface_hub import hf_hub_download
from covt_pilot.io import read_json, write_json

root=Path('/content/covt-pilot')
prior=Path(read_json(root/'runs/latest-colab.json')['run_dir'])
out=root/'runs'/time.strftime('followup-fp32-%Y%m%d-%H%M%S',time.gmtime())
out.mkdir(parents=True,exist_ok=False)
write_json(root/'runs/latest-followup.json',{'run_dir':str(out)})
teacher_info=read_json(prior/'expert_download.json')
teacher=hf_hub_download(teacher_info['model_id'],'depth_anything_v2_vitl.pth',revision=teacher_info['revision'])


def command(*args):
    print('COMMAND',list(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=root,check=True)


write_json(out/'plan.json',{'families':10,'seed':18,'images':40,'calibration_families':2,'pilot_families':8,
    'model_dtype':'float32','attention':'eager','expert_device':'cpu','shuffle_repeats':5,
    'decision':'FP32 chosen after four-image cache precision diagnostic; no E0 gate or tolerance relaxed.',
    'scope':'Exploratory terminal-decoder controls; no answer-path causal claim.'})
command(sys.executable,'-m','covt_pilot','make-scenes','--families',10,'--seed',18,'--out',out/'scenes')
command(sys.executable,'-m','covt_pilot','e0','--manifest',out/'scenes/manifest.jsonl',
    '--audit',prior/'resources/report.json','--decoder-dir',prior/'decoder',
    '--covt-repo',root/'third_party/CoVT','--expert-checkpoint',teacher,
    '--dtype','float32','--device','cuda','--expert-device','cpu','--out',out/'e0')
command(sys.executable,'scripts/state_controls.py','--e0-dir',out/'e0',
    '--decoder-dir',prior/'decoder','--out',out/'controls','--device','cuda','--repeats',5)
print('FOLLOWUP_COMPLETED',str(out),flush=True)
