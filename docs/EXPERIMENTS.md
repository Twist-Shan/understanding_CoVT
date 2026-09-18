# Experiment index

All completed model runs were performed in Colab. The table maps maintained entrypoints to reports and included evidence. The original E0--E5 proposal numbering is not the same as the seven reporting groups.

| Stage | Entrypoint | Findings | Included evidence |
|---|---|---|---|
| BF16 E0 / E1 | `scripts/colab_e0.py` | [Initial run](colab_20260918_results.md) | [initial](../results/2026-09-18/initial/) |
| Precision diagnosis | `scripts/cache_diagnostic.py` | [Precision](cache_precision_diagnostic.md) | Original large archives remain local |
| FP32 / state controls | `scripts/colab_followup.py` | [FP32](colab_20260918_followup.md) | [fp32](../results/2026-09-18/fp32/) |
| Question diagnostics | `scripts/answer_diagnostic.py` | [Prompts](answer_diagnostic_20260918.md) | [answer](../results/2026-09-18/answer/) |
| Geometry shifts | `scripts/shift_validation.py` | [Shifts](shift_validation_20260918.md) | [shift](../results/2026-09-18/shift/) |
| Size balance / intervention | `scripts/balanced_intervention.py` | [Balanced](balanced_intervention_20260918.md) | [balanced](../results/2026-09-18/balanced/) |
| Seven-layer scan | `scripts/layer_scan.py` | [Layers](layer_scan_20260918.md) | [layers](../results/2026-09-18/layers/) |
| Layer blocks / positions | `scripts/block_intervention.py` | [Blocks](block_intervention_20260918.md) | [blocks](../results/2026-09-18/blocks/) |

`results/manifest.json` maps each included evidence file to its original archive path and checksum. Raw results retain original runtime paths. E0 restoration and E1 decomposition are completed pilots; shared-upstream joint interpretation is partial; path localization, directional analysis, and repair remain future work.

Historical report builders under `scripts/build_*.py` consume local archived caches. English notes can be compiled directly from `docs/research/` with the included figures. The Chinese integrated proposal builder additionally expects the original sibling `vlm_proposals/` directory. These historical builders are not needed to run the core package or read the uploaded research notes.
