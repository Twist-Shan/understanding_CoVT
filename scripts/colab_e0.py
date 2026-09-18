"""Run only inside Colab. Small E0, with pinned resources and persistent logs."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REV = "48f5a921c01db42b33ca0949b303b17a3e085bf4"
CODE_REV = "c3497a325b0fdda4d91ade456a01b0e325ee3150"


def main():
    if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
        raise RuntimeError('This runner is restricted to Google Colab; no local experiments.')
    os.chdir(ROOT)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from covt_pilot.io import write_json
    from covt_pilot.resources import audit, recover
    gpu = torch.cuda.get_device_name(0)
    if 'A100' not in gpu:
        raise RuntimeError(f'Expected A100, got {gpu}')
    run = ROOT / 'runs' / time.strftime('colab-e0-%Y%m%d-%H%M%S', time.gmtime())
    run.mkdir(parents=True, exist_ok=False)
    print(f'RUN_DIR={run}', flush=True)
    write_json(ROOT / 'runs/latest-colab.json', {'run_dir': str(run)})

    def command(*args):
        print('\nCOMMAND:', ' '.join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), check=True)

    command(sys.executable, '-m', 'covt_pilot', 'doctor', '--out', run / 'doctor.json')
    print('Auditing pinned model and source', flush=True)
    report = audit(run / 'resources', revision=MODEL_REV, code_revision=CODE_REV)
    if report['status'] != 'tensor_names_present' or report['errors']:
        raise RuntimeError(json.dumps(report['errors']))
    upstream = ROOT / 'third_party/CoVT'
    if not upstream.exists():
        command('git', 'clone', '--filter=blob:none', 'https://github.com/Wakals/CoVT.git', upstream)
    command('git', '-C', upstream, 'checkout', CODE_REV)
    print('Downloading pinned CoVT checkpoint', flush=True)
    snapshot = snapshot_download('Wakals/CoVT-7B-depth', revision=MODEL_REV,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=4)
    recover(run / 'resources/report.json', run / 'decoder', local_checkpoint=snapshot)
    teacher_id = 'depth-anything/Depth-Anything-V2-Large'
    teacher_rev = HfApi().model_info(teacher_id).sha
    write_json(run / 'expert_download.json', {'model_id': teacher_id, 'revision': teacher_rev})
    teacher = hf_hub_download(teacher_id, 'depth_anything_v2_vitl.pth', revision=teacher_rev)
    command(sys.executable, '-m', 'covt_pilot', 'make-scenes', '--families', 3, '--seed', 0,
            '--out', run / 'scenes')
    print('Starting native E0: 3 families / 12 images; one calibration family, two pilot families.', flush=True)
    command(sys.executable, '-m', 'covt_pilot', 'e0', '--manifest', run / 'scenes/manifest.jsonl',
            '--audit', run / 'resources/report.json', '--decoder-dir', run / 'decoder',
            '--covt-repo', upstream, '--expert-checkpoint', teacher,
            '--device', 'cuda', '--expert-device', 'cuda', '--out', run / 'e0')
    print('E0_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
