# CO4 — Evaluation, Limitations, and Social Implications

*Spurious-Correlation Auditor for Chest X-ray Classifiers.* This document evaluates what the
project's measured results actually license as claims, states limitations plainly (including two
claims withdrawn after review), and discusses patient-safety, fairness, and cross-hospital-risk
implications. It assumes the results in CO3.

## 1. What the evidence supports, precisely

It is tempting to summarize a capstone as "the auditor detects, names, and mitigates shortcuts."
The measured evidence supports a narrower, more defensible statement:

- **Detection and localization transfer from synthetic to real confounds.** V2/V3 (synthetic) and
  the tube-reliance/side-consistency measurements in V4b/V5 (real) both succeeded. This part of
  the pipeline generalizes.
- **Naming does not reliably transfer.** V4a passed at a near-ceiling bar on an unambiguous
  synthetic confound; V4b failed on a real device confound, specifically by *fabricating* devices
  on clean images (control-hallucination 0.50) — the failure mode the project's precision-over-
  recall design was built to catch, and it caught it. This is a **bounded capability**, not a
  general one: naming should be trusted for salient, unambiguous shortcuts, and should not be
  trusted for subtle real-world confounds without further validation (see §4).
- **Mitigation with no target labels works on a real confound.** V5's tube result (worst-group
  0.67→0.83, ID preserved) used no test-distribution labels and is corroborated by two
  independent methods (JTT and DFR) agreeing within a few points, well outside seed variance.
  This is the project's most defensible quantitative result.
- **Mitigation does not obviously transfer across a real distribution shift.** V6 shows a real
  19-point cross-site AUC drop, but the zero-shot, label-free test of whether reducing a known
  source shortcut narrows that drop returned a null result. **The project does not show that its
  mitigation method fixes cross-site generalization gaps** — only that a real gap exists and that
  a specific attempt to explain it via one known shortcut did not pan out.
- **A shortcut-reliant backbone retains real, if degraded, task-relevant features.** The linear
  probe (0.690 vs. chance 0.50, clean 0.845) means that even a strongly shortcut-reliant model is
  not "starting from nothing" for mitigation purposes — a practically useful fact for anyone
  choosing between retraining from scratch and last-layer correction.

## 2. Two claims withdrawn — why this matters more than a clean narrative would

Two claims were written into the project record before the control that tests them existed, and
both were corrected once that control was run:

1. **"Pure-shortcut model is hollow"** → withdrawn after the linear probe measured AUC 0.690, not
   near-chance. The project's own prior measurement (tag-free accuracy of 0.74 during V2) already
   argued against "hollow"; the claim was written anyway, as narrative color rather than a
   measured statement.
2. **"Mitigation validated across three settings"** → withdrawn after the zero-shot source-DFR
   test on V6 returned a null result. The "third setting" (cross-site) had in fact only been
   tested with target labels present (supervised adaptation), which does not test a mitigation
   hypothesis at all — it would help even with no shortcut in play.

Both errors were the same failure mode: asserting an interpretation the experiment did not
license, discovered by asking "what specific control would falsify this?" and then running it,
rather than accepting the more convenient number. We report both corrections in full — including
in CO1's novelty framing and CO3's spike record — because a wrong claim that survives to a final
document is a worse outcome for a capstone than a smaller, correct one. The broader lesson (now
saved as a standing practice for this project) is: **before writing an interpretive claim, name
the control that could falsify it, and run that control first.**

## 3. Comparison to the closest prior/concurrent art (see CO1 for full citations)

- Against **RoentMod**: no generative model, no fine-tuning, dramatically lower compute (a free
  API pass over ~200 crops vs. a diffusion pipeline) — but RoentMod's correction-only design never
  had to clear the bar our naming step failed at (V4b), so this is a genuine trade rather than a
  strict improvement.
- Against the **concurrent chest-drain-mitigation paper** (arXiv:2604.17110, May 2026): both teams
  independently targeted the identical real shortcut and both found it partially mitigable,
  by different mechanisms and different metrics — mutually corroborating rather than competing.
- Against **general-domain VLM auditing** (SpurLens, RaVL): this project is the only one found
  operating in the medical/CXR setting with a control-hallucination-gated metric, but it is also
  the only one found reporting a hard failure of the naming step on a realistic confound — a data
  point those general-domain papers, evaluated on natural images, did not have occasion to surface.

## 4. Limitations

**Naming reliability is bounded, and the boundary is not yet well characterized.** V4a and V4b
are two points, not a curve — we know naming works on an unambiguous synthetic marker and fails
on a subtle real device, but we do not know where between those extremes it starts to fail, or
whether a different VLM, prompt, or crop resolution would shift that boundary. The project
deliberately did not attempt to rescue V4b's precision/recall with prompt engineering, because
CLAUDE.md's own design specified that a V4b failure (after V4a passed) triggers a pivot rather
than iterative rescue — a principled stopping rule, but one that leaves the boundary
uncharacterized rather than mapped.

**The auditor's own VLM component was not tested for image-independence.** A separate 2026 paper
(arXiv:2606.17710, CO1 §2e) shows VLMs can answer chest-X-ray questions correctly from language
priors alone, without attending to the image. If Llama-4-Scout's V4a/V4b descriptions were
partly driven by generic radiology-report phrasing rather than the crop's actual content, our
precision/recall numbers would still be accurate (they're scored against ground truth), but the
*interpretation* — "the VLM saw and named the confound" — would be weaker than "the VLM's output
happened to match the confound." We did not run the counterfactual-image test that would
distinguish these, and flag it as the most direct follow-up to strengthen V4a's interpretation.

**Non-expert ground truth for the real device confound.** NEATX's chest-drain annotations come
from two data-science students, not radiologists, with documented "moderate to almost perfect"
agreement against expert labels in the source paper — good enough to build a training-time
enrichment signal and a V5 mitigation target, but a noisier ground truth than a radiologist-drawn
segmentation (e.g., CANDID-PTX, which requires a completed ethics-training course we did not have
time to obtain) would provide for V4b's precision/recall scoring specifically.

**V6's causal attribution is genuinely unresolved.** As stated in CO3 §10, the cross-site drop's
cause is not established by this project's design. This is not a limitation we can currently
close within scope; a proper next step (isolating scanner effects, re-labeling PadChest with the
NIH labeler, or a multi-site study with more than two sites) is future work, not something this
capstone can retrofit.

**Single-seed sweep; 3-seed baseline table.** The correlation-strength sweep (CO3 §8) is reported
at one seed per point due to compute budget on a single consumer GPU; the trend (monotonic
reliance growth) is clear enough to be informative, but individual point values could shift
somewhat under a different seed. The baseline comparison table (ERM/JTT/DFR) does carry proper
3-seed variance and is the number we would defend more strongly.

**Hardware-constrained scale.** All training ran on a single RTX 3050 (6GB), which shaped batch
size (16, after an OOM at 32), image resolution (224px, standard for this literature), and the
decision to work with capped/subset data for several spikes (e.g., the 7,572-image PadChest
working set rather than the full 160,861). None of the measured results depend on the hardware
in a way that would not reproduce at larger scale — but larger-scale replication was out of reach
for this project and is not claimed.

## 5. Patient safety

The project's central patient-safety argument is illustrated, not just asserted, by its own
results: a pneumothorax classifier that keys on the chest tube used to *treat* pneumothorax will
silently fail exactly on patients who have not yet been treated — the population where correct
detection matters most. V5's tube-reliance measurement makes this concrete: without a tube, the
unmitigated model's accuracy on genuine pneumothorax cases was **0.67**, a clinically unacceptable
miss rate for the population the model exists to help, recovered to **0.83** by mitigation with
no additional labels required at deployment time. The project also demonstrates a failure mode
specific to an *automated* auditor: V4b shows that naively trusting a VLM's shortcut description
is itself a patient-safety risk, since it fabricated devices on 50% of clean images — an auditor
that hallucinates findings is not a safe substitute for the manual review it aims to reduce, which
is precisely why the project's own gate design treats a false alarm as more costly than a miss.

## 6. Cross-hospital risk and fairness

V6's confirmed 19-point AUC drop from a US hospital dataset to a Spanish one is itself a fairness
and equity finding independent of any shortcut-mitigation question: a model validated at one
institution and deployed at another, without re-validation, will underperform for the second
institution's patient population in a way this project measured directly rather than assumed from
the literature. That the specific mechanism (chest tube shortcut) was ruled out as the sole
explanation is itself clinically relevant: it means institutions cannot assume that "fixing the
known shortcuts" is sufficient due diligence before cross-site deployment — the residual gap after
a targeted intervention was, in this case, almost the entire original gap. Combined with the
non-expert-label limitation (§4), this project supports a conservative recommendation: cross-site
performance drops should be treated as requiring site-specific validation, not resolved by a
single mitigation technique applied once at the source institution.

## 7. What we would do differently, and what we would keep

**Keep:** the validation-gated spike structure, which caught a real problem (V4b) at the point it
was measured rather than after a full system was built around an unvalidated naming assumption;
the discipline (once corrected) of running the specific falsifying control before writing an
interpretive claim; building the confound-agnostic core (CO2 §1) that let one DFR/Grad-CAM
implementation serve three different confound/site settings without duplication.

**Would do differently:** run the linear probe and the zero-shot source-DFR test *before* writing
the narrative claims they ended up correcting, rather than after; budget time to attempt the
CANDID-PTX ethics-training pathway earlier, for radiologist-grade real-confound ground truth in
V4b/V5 rather than relying solely on non-expert NEATX labels; test the auditor's own VLM for
image-independence (§4) as part of V4, not as a noted-but-unexecuted follow-up.
