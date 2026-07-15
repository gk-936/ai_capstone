| opacity | model | precision | recall | control-hallucination |
|---|---|---|---|---|
| 0.25 | scout | 0.600 | 0.030 | 0.020 |
| 0.25 | pixtral | 0.420 | 0.210 | 0.290 |
| 0.5 | scout | 0.963 | 0.520 | 0.020 |
| 0.5 | pixtral | 0.588 | 0.400 | 0.280 |
| 0.75 | scout | 0.978 | 0.900 | 0.020 |
| 0.75 | pixtral | 0.556 | 0.400 | 0.320 |
| 1.0 | scout | 0.978 | 0.900 | 0.020 |
| 1.0 | pixtral | 0.626 | 0.520 | 0.310 |

## Addendum — 2026-07-14, working thesis update

**Conclusion, confirmed by contact sheet + numbers**: opacity-driven subtlety produces a
recall cliff (threshold between 0.25→0.5), NOT a hallucination cliff. Control-hallucination
is flat across all four opacity levels for both models (scout ~0.02 throughout; pixtral
~0.28-0.32 throughout). This is the opposite of what would be needed to mechanistically
bridge V4a's 0.02 hallucination to V4b's 0.50-0.83.

- **Original thesis (pre-Exp1)**: "VLM fabrication rises as a function of visual
  subtlety/faintness of the confound."
  **Status: NOT SUPPORTED on the opacity axis. Ruled out by Exp 1.**

- **Updated working thesis (going into Exp 2)**: "VLM fabrication is driven by the
  anatomical plausibility of the confound category, not by how visible it is. A synthetic
  tag is never mistakable for anatomy at any opacity — either seen clearly or not seen at
  all, with no ambiguous middle ground. A real device sits in a space of plausible
  near-misses (other lines/tubes/catheters, or normal anatomy) that a VLM's medical-image
  priors can generate boilerplate about regardless of what's actually in the crop."

This is still a working thesis, not a conclusion — Experiment 2 (image-independence test)
is the direct test of it.