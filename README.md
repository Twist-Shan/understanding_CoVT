# understanding_CoVT

**Do CoVT's depth visualizations reflect the information used to answer?**

This repository studies the released **Chain-of-Visual-Thought (CoVT)** depth model. It restores the original depth readout, separates the contributions of CoVT states and expert image features, and tests whether internal-state interventions affect generated answers. It contains exploratory experiments, not a newly trained model.

## Start here

- [Research notes (PDF)](docs/research/CoVT_research_notes.pdf): beginner-friendly model explanation, exact prompts, experiments, figures, and the revised research plan.
- [Colab reproduction guide](docs/COLAB.md): environment, stage order, prerequisites, and remote-only execution.
- [Experiment index](docs/EXPERIMENTS.md): completed experiments and their evidence.
- [Original detailed instructions (Chinese)](docs/reproduction_zh.md).
- [Machine-readable results](results/README.md): selected original records with SHA-256 provenance.

## What we have observed

| Experiment | Observation | Scope |
|---|---|---|
| FP32 native restoration | 40/40 passed checks; depth 32/32, text 8/32 on evaluation images | 10 synthetic scene families, 2 for calibration |
| Decoder input swaps | State-only: 0/8 relation flips; expert-feature-only: 8/8 | Decoder branch; no answer intervention |
| Fixed-state readout | 199/200 correct, 1 abstention on geometry shifts | Synthetic images; expert features remain image-dependent |
| Size-balanced questions | Text 25/32; fixed-state depth 30/32 | 8 related scene families |
| Layer / position expansion | 200 interventions preserved generated token sequences | 20 selected targets, 6 families; up to 28 layers and 15 thought positions |

These results show that a useful conditional depth visualization does not by itself establish how the answer was formed. We have **not** established universal state non-use, localized a defective answer reader, or performed repair training. The shuffled control permutes the **four state slots**, not spatial locations in expert features. See the notes for alternative explanations and failure counts.

## Repository layout

```text
src/covt_pilot/      Native readout, model wrapper, scene geometry, metrics
scripts/            Colab experiment entrypoints, audits, and report builders
tests/              Unit tests and mocked integration tests
results/            Small archived JSON/JSONL records and checksums
docs/research/      Notes PDF, editable LaTeX, and attributed figures
docs/               Protocols, findings, and reproduction instructions
.github/workflows/  Remote CPU software checks
```

The Python package and command remain `covt-pilot` to preserve existing scripts. Clone the repository as `/content/covt-pilot` in Colab; historical staged runners use that path.

## Reproduction

**Run model experiments on Colab A100, not the local workstation.** Follow [COLAB.md](docs/COLAB.md) before launching any stage. Staged scripts require earlier run artifacts; they are not independent one-click jobs. The initial E0 script reproduces the historical BF16 pilot; the FP32 follow-up is a separate stage.

```bash
# On a configured remote Python 3.11 runtime:
python -m pip install -e '.[model,test]'
python -m pytest -q
python scripts/colab_e0.py
python scripts/cache_diagnostic.py
python scripts/colab_followup.py
```

No new model experiments run automatically on push. GitHub Actions executes CPU software tests and synthetic plumbing checks, without downloading the VLM or depth expert.

## Data and artifacts

The archived experiments contain **284 base-image records in 71 scene families** across four batches. Related variants and repeated prompts/interventions are not independent samples. `results/` includes selected original records, plans, calibration and audit summaries. Historical `/content/...` paths are preserved as provenance; linked tensor caches are intentionally absent.

Full activation caches, checkpoints, runtime notebooks, generated HTML galleries, and downloaded archives remain in local `runs/` or `output/` and are excluded from Git. A fresh clone can read the notes and audit the included records. Rebuilding every plot or replaying interventions additionally requires the original caches or fresh remote runs; compact JSON alone is insufficient.

## Sources and rights

- [Official CoVT repository](https://github.com/Wakals/CoVT)
- [Official paper](https://wakalsprojectpage.github.io/covt-website/static/pdf/paper.pdf)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

The pinned CoVT model and code revisions are documented in [the reproduction guide](docs/reproduction_zh.md). Upstream figures, code, and weights retain their original rights and license terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This research snapshot does not assign a new license to third-party material.
