# Reproduce on Colab A100

Model experiments are remote-only. Choose an A100 runtime and a clean Python 3.11 environment. The initial runner checks for Colab and A100. FP32 generation has substantial memory requirements; successful pilot runs on A100 40 GB do not guarantee that arbitrary longer inputs fit.

## Checkout and environment

Authenticate to this private GitHub repository using your own GitHub credentials in the remote environment; never paste a token into a committed notebook or URL. Once Git is authenticated:

```bash
git clone https://github.com/Twist-Shan/understanding_CoVT.git /content/covt-pilot
cd /content/covt-pilot
python --version  # use Python 3.11
python -m pip install -e '.[model,test]'
python -m pytest -q
```

Use the selected environment's Python for every command. If Colab's default Python differs, create a separate 3.11 environment first. Do not silently substitute the current Transformers release: the wrapper requires `transformers==4.50.1`, with the model extra pinning Torch 2.5.1 / torchvision 0.20.1. `report` is an optional dependency extra for local document processing, not model execution.

## Required stage order

Run one command at a time and inspect its audit before proceeding. Each stage creates a timestamped directory and writes a `runs/latest-*.json` pointer. A new Colab session needs the actual earlier artifacts restored, not only these pointer files.

1. `python scripts/colab_e0.py`: downloads pinned resources, restores the native decoder, prepares the expert, and reproduces the initial BF16 E0/E1. Writes `latest-colab.json`.
2. `python scripts/cache_diagnostic.py`: compares precision/cache conditions using the initial artifacts.
3. `python scripts/colab_followup.py`: FP32 E0/E1 and state controls. Requires the initial decoder and expert provenance; writes `latest-followup.json`.
4. Optional branches after step 3: answer diagnostics, geometry shifts, or the balanced intervention below.

### Answer diagnostics

This script expects `runs/backbone-revision.json` in addition to the two earlier pointers. Recover the historical model identifier and immutable revision from `results/2026-09-18/answer/plan.json` under `models.backbone`; do not resolve a new latest revision while claiming an exact reproduction.

```python
import json
from pathlib import Path
plan = json.loads(Path('results/2026-09-18/answer/plan.json').read_text())
Path('runs/backbone-revision.json').write_text(json.dumps(plan['models']['backbone'], indent=2))
```

Then run `python scripts/answer_diagnostic.py`.

### Geometry shift branch

```bash
python scripts/shift_validation.py expert
python scripts/shift_validation.py native
```

The expert stage creates the scenes and frozen-readout outputs; the native stage resumes that run. `scripts/summarize_validation.py` also expects the answer-diagnostic branch to exist.

### Balanced and intervention branch

```bash
python scripts/balanced_intervention.py
python scripts/summarize_balanced.py
python scripts/layer_scan.py
python scripts/summarize_layer_scan.py
python scripts/block_intervention.py
python scripts/summarize_block_intervention.py
```

Layer and block scans consume the balanced run's recorded native states and eligible donor matching. The block script conditionally extends to all thought-prefix positions. These are exploratory, fixed pilot protocols; changing a selection rule creates a new experiment and should be recorded.

## Preserve outputs before disconnecting

Download generated archives and run metadata, verify checksums, and preserve the pinned resource identifiers. Weights and state/feature caches are excluded from Git. The included `results/` is a compact evidence snapshot, not a drop-in replacement for all `runs/` artifacts. Do not run GPU scripts merely to build the PDF.

## Software validation

GitHub Actions runs unit tests and mocked integration tests on a remote CPU runner, plus the synthetic smoke command. It does not execute the above model experiments. Local repository preparation may parse source and verify checksums without loading a model.
