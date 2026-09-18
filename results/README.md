# Archived evidence

`2026-09-18/` contains 157 selected, unmodified JSON/JSONL files from seven archived Colab run groups. `manifest.json` records each original source path, byte count, and SHA-256. Verify with:

```bash
python scripts/verify_results.py
```

Paths inside records describe the original Colab environment. Model weights, activation/feature caches, scene images, and downloaded archives are not included. An `artifact` field can therefore point to a file available only in the full local archive. No metrics were recomputed by the export. Prompt text and synthetic model outputs are retained for audit.

Read [the experiment index](../docs/EXPERIMENTS.md) and [research notes](../docs/research/CoVT_research_notes.pdf) before interpreting results. The seven groups reuse related images; counts of interventions are not counts of independent scenes.
