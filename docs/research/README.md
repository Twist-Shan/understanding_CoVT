# Research notes

Start with [the PDF](CoVT_research_notes.pdf). The title states the research question:
**Do CoVT's Depth Visualizations Reflect the Information Used to Answer?**

The notes introduce VLMs and CoVT, explain exact prompts and metrics, report seven groups of pilot experiments, and distinguish results from planned causal tests.

Compile the editable source with `latexmk -pdf CoVT_research_notes.tex` in this directory. Figure assets are included, so compilation needs neither model weights nor raw caches. Reproduced original-paper figures retain the authors' rights; see `paper_figure_provenance.json` and the citations in the notes.

Correction during repository preparation: the shuffled control permutes the four **CoVT states**, not expert spatial features. Results are unchanged; the explanatory wording has been corrected against `src/covt_pilot/controls.py` and `scripts/shift_validation.py`.
