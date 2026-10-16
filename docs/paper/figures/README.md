# Paper figures — index

Canonical folder for project paper figures.
The capstone writeup lives at `docs/capstone_writeup.md`.

## What each figure shows (and where its numbers come from)

| File | What it shows | Source data |
|---|---|---|
| `fig0_architecture.pdf` / `.png` | Detect→localize→name→mitigate pipeline schematic, with the isolated VLM-naming module boundary called out | Methods overview (no experiment re-run) |
| `figure1_naming_boundary.pdf` / `.png` | The naming boundary: control-hallucination on tube/tag-FREE crops, V4a vs. V4b | §5.1/5.2 — Scout & Pixtral |
| `auditor_before_after_tube.png` | Grad-CAM before/after DFR on the real tube model | §5.3 — demo figure from the auditor loop |
| `fig3_overname_underrname_roster.pdf` / `.png` | Control-hallucination across all 4 tested models (Scout, Pixtral, qwen2.5-vl-7b, CheXagent-2-3B), V4a vs. V4b | §6.3 — extends Fig. 1 to the full roster |
| `fig4_occlusion_grounding.pdf` / `.png` | Image-independence/occlusion check: pos→neg vs. neg→pos rates, relevant vs. irrelevant occlusion, V4a vs. V4b | §6.2 — grouped-bar form of the occlusion table |
| `fig5_crosssite_forest.pdf` / `.png` | Forest plot: NIH→PadChest drop, supervised adaptation, zero-shot source-DFR, each with its 95% bootstrap CI where reported | §5.3 / Appendix B — the "CI includes zero" null |
| `fig6_mitigation_comparison.pdf` / `.png` | ERM/JTT/DFR worst-group vs. in-distribution accuracy, tube vs. tag, 3-seed mean ± SD | §5.3 — the mitigation table |
| `exp1_difficulty_curve.png` | Opacity sweep: precision/recall/hallucination vs. tag opacity ("recall cliff, not hallucination cliff") | §6.1 — Exp-1 sweep tooling |
| `exp1_contact_sheet.png` | Example crops at each opacity level (qualitative) | §6.1 — same sweep, supporting evidence |
| `sweep_figure.png` | Correlation-strength sweep: reliance gap vs. tag correlation *p* (0.6→0.95) | §5 — correlation-sweep results |

## Regenerating

**The six `fig0 / figure1 / fig3–fig6` pairs** (each as `.png` for preview/Word and
`.pdf` vector for LaTeX) are data-driven outputs of:
```bash
python docs/paper/figures/generate_new_figures.py
```
matplotlib only (`Agg` backend) — no experiment re-run. Every number is hardcoded
from already-published results in `docs/capstone_writeup.md`; the script's own header/source comments say
exactly which section each number traces to. Outputs land in this same folder.

**The other four** (`auditor_before_after_tube.png`, `exp1_difficulty_curve.png`,
`exp1_contact_sheet.png`, `sweep_figure.png`) are **not** produced by that script —
they come from the experiment/demo tooling (auditor-loop demo figure, Exp-1 sweep
plotting/contact-sheet code). Do not expect the script to refresh them.

## Caveats (read before reusing a figure standalone)

- `fig3`: qwen2.5-vl-7b has no reported V4a value in the source material — left
  blank and labeled "not reported", never invented. CheXagent's V4b = 0.0 is
  annotated as a taxonomy gap (no device vocabulary), not a pass.
- `fig4`: the V4a `relevant`-occlusion condition is contaminated by the gray
  occluder itself tripping the `names_confound` matcher (documented in §6.2);
  the grounding claim rests on the V4b-internal contrast only. The charts no
  longer carry footnotes, so pair this figure with the §6.2 text.
- `fig5`: supervised target adaptation has no CI in the source material (open
  diamond marker, no whiskers) — don't add error bars without a real bootstrap
  re-run (see `docs/writeup/statistical_hardening.md`).
- `figure1`: bars are control-hallucination rates on confound-FREE crops
  (n=100/cell); whiskers are 95% bootstrap CIs (10k resamples, seed 1337).
  Synthetic controls sit near 2% (Scout); real-tube controls jump to 55–83%.
