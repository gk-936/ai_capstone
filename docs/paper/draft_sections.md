# Paper draft — sections

Working draft, paper register. Every number below is traceable to a document under
`docs/writeup/` or `PROGRESS.md`. Where a number a section wants does not exist in those
sources, it is marked `[NUMBER NOT IN SOURCES]` rather than estimated. Citations are given by
author/identifier only; no venue formatting or page numbers are asserted here.

---

## 1. Methods

[CITATIONS — PENDING]

### 1.1 Overview

We audit a chest X-ray classifier for reliance on spurious shortcuts through a four-stage loop:
**detect** that the classifier's accuracy depends on a confound, **localize** the confound via
attribution, **name** it in clinical language using a general-purpose vision-language model
(VLM) that is never told what to look for, and **mitigate** the reliance by last-layer
retraining. The loop is evaluated on a synthetic confound, a real device confound, and a real
second site. The study was executed as six validation-gated spikes (V1–V6), each with a
pre-registered pass/fail threshold and a stop-and-report point, so that a failed stage
terminates a claim rather than being iterated around.

### 1.2 Data and model

The primary corpus is NIH ChestX-ray14 (112,120 images, resized to 224x224), with
patient-disjoint train/validation/test splits (73k/13k/25.6k) built from the official split
lists; image-level splitting is never used, as it leaks a patient's other studies across
splits. The target pathology is pneumothorax (roughly 3–5% prevalence depending on split). The
classifier is an ImageNet-pretrained DenseNet-121 trained with automatic mixed precision, batch
size 16, and a `pos_weight`-corrected binary cross-entropy loss for the class imbalance.

Real chest-drain ground truth comes from **NEATX** (Cheplygina et al., arXiv:2309.02244), which
supplies non-expert chest-drain annotations for a subset of ChestX-ray14 and PadChest. The
second site is **PadChest** (Hospital San Juan, Spain), used in its 224x224 release; a
7,572-image working subset (573 pneumothorax-positive, 7,000 negative) was extracted under a
disk constraint.

### 1.3 V1 — pipeline sanity

A baseline pneumothorax classifier is trained end-to-end and gated on in-distribution test AUC
>= 0.75. This spike exists solely to establish that any subsequent degradation is attributable
to an intervention rather than to a broken pipeline.

### 1.4 V2 — shortcut induction

A synthetic confound — a 20x20 opaque corner tag — is stamped onto training images with
correlation strength *p* to the pneumothorax label. Reliance is quantified as the **reliance
gap**: accuracy with the confound aligned as trained, minus accuracy with the confound removed,
measured on a class-balanced subset. A third condition (tag *flipped*, i.e. anti-correlated)
provides a lower bound. The gate is a gap > 20 percentage points. A companion
**correlation-strength sweep** trains the same architecture at *p* in {0.6, 0.7, 0.8, 0.9,
0.95} to characterize how reliance and its mitigability scale with confound strength.

### 1.5 V3 — attribution localizes the confound

Grad-CAM is computed on the shortcut-reliant checkpoint. Localization is scored with a pointing
game (`peak_in_region`) against a generic box `(y0, x0, y1, x1)`, plus a `mass_fraction`
statistic. Crucially, the same images with the confound *removed* serve as a control, which
rules out "that corner is always salient" as an alternative explanation. All attribution
geometry is confound-agnostic: no function knows whether the box contains a synthetic tag or a
real device, so the identical code serves the synthetic and real settings.

### 1.6 V4 — blind VLM naming

Salient crops from confound-positive images, plus confound-free control crops, are exported
with a manifest carrying exact `has_confound` labels the VLM never sees. Crops are captioned by
a general-purpose, non-fine-tuned VLM under a **blind prompt** that never mentions the target
("describe anything artificial, non-anatomical, or out of place... if normal, say so").
Free-text descriptions are scored by conservative keyword matchers — `names_confound` for the
synthetic-marker vocabulary, `names_tube` for device-specific vocabulary (tube / drain /
catheter / line / wire) — into precision, recall, and **control-hallucination**: the rate at
which a device or marker is named on a confound-free crop. Matchers are unit-tested not to fire
on generic anatomy language.

The naming stage is deliberately split into two gates with different bars, because they test
different things:

- **V4a (synthetic tag)** proves the *method* works where ground truth is exact and the
  confound is unambiguous. Gate: precision >= 0.90, recall >= 0.90, control-hallucination < 10%.
- **V4b (real chest tube)** proves the method *transfers*. Gate: precision >= 0.80, recall >=
  0.70, control-hallucination < 10%.

Precision is gated above recall by design: for an auditor, a fabricated confound costs more
than a missed one, since a tool that invents shortcuts is worse than no tool. The
control-hallucination bar is measured separately and is the hardest to relax. The naming stage
is isolated behind a file contract (`crops/ + manifest.json -> predictions.jsonl`), so the
provider is interchangeable and no downstream computation depends on it.

Two general-purpose VLMs were run through the identical pipeline on the identical crops:
**Llama-4-Scout** and **Pixtral-12B**. A third (Gemini) was not run to completion because its
free tier caps daily requests per model below the 200 crops required; it is not reported as a
result.

### 1.7 V5 — mitigation

Mitigation is **Deep Feature Reweighting (DFR)**: extract frozen penultimate features, draw a
group-balanced sample over `(confound, label)` groups, fit a logistic head on standardized
features, and swap only the final linear layer. Group balancing handles the structurally empty
group (chest tubes co-occur almost exclusively with pneumothorax-positives, so `(tube=1,
label=0)` does not exist). **ERM** and **JTT** (upweighting the frozen model's own errors and
refitting the head) are run from the same feature-extraction path as baselines, over 3 seeds.
Because none of these three methods import the naming module, this comparison doubles as the
**naming-off ablation**: the loop's mitigation result cannot depend on the VLM. Metrics are
worst-group accuracy (pneumothorax *without* the confound), in-distribution accuracy
(pneumothorax *with* the confound), and the reliance gap.

A **linear probe** control freezes a shortcut-trained backbone and fits a fresh head on
confound-free images, to test directly whether shortcut reliance hollows out or merely degrades
core-feature learning.

### 1.8 V6 — cross-site generalization

The V1 model is evaluated unchanged on PadChest. Two mitigation variants are distinguished, and
the distinction is the point of the spike:

- **Supervised target adaptation** — fit the last layer on *labeled* PadChest data. This tests
  adaptation, not shortcut mitigation: it would help even with no shortcut present.
- **Zero-shot source-DFR** — reduce the known source-side shortcut using NIH data only, touch
  **no** target labels, then evaluate PadChest. This is the test that can license a claim that
  the cross-site gap is shortcut-driven.

PadChest images are 16-bit while NIH images are 8-bit. Reconciling this at load time (rather
than by changing the model's evaluation transform) is required for validity; the NIH reference
AUC is re-measured after every such change and serves as the control that validates any
cross-site number.

### 1.9 Reproducibility

Every run is config-driven (typed YAML with inheritance and CLI overrides), fixed-seeded, and
logged to a timestamped directory containing the resolved config, `metrics.json`, and a
checkpoint. Dataset paths are configuration values; no image or checkpoint is version
controlled.

---

## 2. Results — the central finding

[CITATIONS — PENDING]

The classifier trains to a sane baseline (V1: test AUC **0.8448**, validation 0.8269 — within
the published NIH-pneumothorax range) and cheats measurably when given the opportunity (V2:
reliance gap **44pp**; aligned 0.94, removed 0.50, flipped 0.05). The removed-accuracy landing
at exactly 0.500 is the pure-shortcut fingerprint, predicted and verified with a synthetic
tag-detector before the run rather than rationalized after it. Attribution localizes the
confound essentially perfectly (V3: peak-in-box **100%** on tagged positives vs **0.5%** on the
same images untagged; mass fraction 0.14 in a box covering 0.8% of image area).

### 2.1 Naming succeeds on the synthetic confound and fails on the real one

**V4a passes at near-ceiling.** Blind-prompted Llama-4-Scout named the injected tag with
precision **0.978**, recall **0.90**, and control-hallucination **0.02**, over 100 tagged
positives and 100 confound-free controls. Recall landed exactly at its 0.90 bar on ten genuine
model misses — an easy target already producing misses, which foreshadowed the harder case. The
keyword matcher was corrected mid-analysis after a manual audit of all 23 original misses found
that roughly 13 were the VLM correctly describing a "rectangular / opaque white object" in
phrasing the initial regex did not cover; the fix was verified not to have relaxed the
control-hallucination check, which remained at 0.02.

**V4b fails, and fails specifically on fabrication.** On real chest-tube crops the same model,
same prompt, same pipeline yielded precision **0.57**, recall **0.66**, control-hallucination
**0.50**. Detection and localization still worked on the real confound (tube reliance gap
**28.75pp**; Grad-CAM side-consistency **96.6%**) — only naming broke. An audit of the control
crops found **47 of 50** false positives explicitly naming a device ("endotracheal tube,"
"central line," "catheter") on images containing none: genuine hallucination, not a scorer
artifact.

This is the pre-registered pivot condition (V4a passes, V4b fails), and it was executed as
specified rather than iterated around: automated naming is withdrawn as a validated capability
for real device confounds, and V4b is reported as a bounded negative result.

### 2.2 The failure is a property of general VLMs, not of one model

The obvious objection to a single-model negative is that a different VLM would not fail. We
re-ran the **identical** 200 V4b crops, the **identical** blind prompt, and the **identical**
`names_tube` matcher through two independent VLM families:

| model | precision | recall | control-hallucination |
|---|---|---|---|
| Llama-4-Scout | 0.545 | 0.660 | **0.550** |
| Pixtral-12B | 0.494 | 0.810 | **0.830** |

Both fabricate a device on a *majority* of clean control crops. The Scout re-run (0.550)
reproduces the original V4b value (0.500) within run-to-run API and decoding drift; Pixtral is
markedly worse. Audit of the raw text: Scout invents specific devices on clean anatomy
("endotracheal tube," "central venous catheter"); Pixtral emits near-identical boilerplate
("linear radiopaque object... looks like a medical device or foreign body") on **83 of 100**
control crops. The matcher is conservative on positives — it does not fire on hedged
"linear artifact" phrasing — which depresses recall in the safe direction rather than inflating
hallucination.

An auditor built on off-the-shelf general-purpose VLMs would therefore raise a fabricated
shortcut alarm on **55–83%** of clean images. This is precisely the false-alarm cost the
precision-over-recall gate was designed to catch, now established as a cross-model property.
It is independently corroborated by HalluCXR (Wang & Li, arXiv:2605.20469), which reports
61.9–82.3% overall hallucination across six VLMs on 856 chest radiographs, with severity
*highest on normal images* — the same qualitative signature, in a different model population
on a different task.

### 2.3 What survives the naming failure

Mitigation, which does not depend on naming, holds on the real confound. DFR moves worst-group
(pneumothorax-without-tube) accuracy **0.67 -> 0.79 (3-seed; single-draw run gives 0.83)** while in-distribution
(pneumothorax-with-tube) accuracy is preserved **0.96 -> 0.96**, narrowing the reliance gap
**0.29 -> 0.17**, using no test-distribution labels. Across 3 seeds, two independent methods
agree and clear the ERM floor by roughly 12 points, far outside seed variance:

| confound | method | worst-group | ID | reliance gap |
|---|---|---|---|---|
| tube | ERM | 0.672 ± 0.022 | 0.959 ± 0.002 | 0.287 ± 0.025 |
| tube | JTT | 0.791 ± 0.047 | 0.940 ± 0.019 | 0.149 ± 0.031 |
| tube | DFR | 0.790 ± 0.036 | 0.964 ± 0.007 | 0.174 ± 0.029 |
| tag | ERM | 0.500 ± 0.000 | 0.940 ± 0.000 | 0.440 ± 0.000 |
| tag | JTT | 0.608 ± 0.018 | 0.531 ± 0.016 | −0.077 ± 0.025 |
| tag | DFR | 0.609 ± 0.022 | 0.603 ± 0.014 | −0.006 ± 0.024 |

The synthetic tag tells the cautionary half of the story: the gap closes (0.44 -> 0.02) but
in-distribution accuracy falls 0.94 -> 0.62. A linear probe run specifically to falsify the
tempting "the backbone is hollow" reading measured probe AUC **0.690** — above chance (0.50),
below the clean baseline (0.845). Shortcut reliance **degrades but does not eliminate**
core-feature learning, and the degradation is strength-dependent.

Cross-site, a real gap exists and is not explained by the shortcut. NIH reference AUC
**0.842** falls to PadChest **0.650**, a drop of **0.192**, matching the magnitude reported by
Zech et al. Supervised target adaptation recovers **+0.082** — but that is last-layer fitting on
*labeled* PadChest and would help absent any shortcut, so it licenses no mitigation claim. The
honest test, zero-shot source-DFR with no target labels touched, returns **+0.005: null**.
Reducing a known real source-side shortcut does not improve cross-site transfer. The 0.192 drop
is distribution shift whose cause this design cannot identify; shortcuts are one unproven
candidate among scanner, population, label-pipeline, residual-intensity, and subset-composition
differences.

---

## 3. Results — mechanism

[CITATIONS — PENDING]

V4a and V4b are two points, not a mechanism. Two follow-up experiments ask *why* naming breaks:
whether fabrication is driven by the confound's **visibility**, and whether it is grounded in
the image at all.

### 3.1 Experiment 1 — the difficulty curve is a recall cliff, not a hallucination cliff

The intuitive hypothesis is that fabrication rises as the confound gets fainter. We tested it
directly by sweeping the synthetic tag's opacity across {0.25, 0.5, 0.75, 1.0} and re-scoring
both models with the unchanged pipeline (n = 200 crops per cell, 100 positive + 100 control):

| opacity | model | precision | recall | control-hallucination |
|---|---|---|---|---|
| 0.25 | scout | 0.600 | 0.030 | 0.020 |
| 0.25 | pixtral | 0.420 | 0.210 | 0.290 |
| 0.50 | scout | 0.963 | 0.520 | 0.020 |
| 0.50 | pixtral | 0.588 | 0.400 | 0.280 |
| 0.75 | scout | 0.978 | 0.900 | 0.020 |
| 0.75 | pixtral | 0.556 | 0.400 | 0.320 |
| 1.00 | scout | 0.978 | 0.900 | 0.020 |
| 1.00 | pixtral | 0.626 | 0.520 | 0.310 |

Reducing visibility produces a **recall cliff** — Scout's recall collapses 0.90 -> 0.52 -> 0.03
between opacity 0.75, 0.5 and 0.25 — while **control-hallucination stays flat**: ~0.02 for
Scout and ~0.28–0.32 for Pixtral across *every* opacity level. Faintness makes the model miss
the confound; it does not make the model invent one.

This **rules out visibility as the bridge** from V4a's 0.02 hallucination to V4b's 0.50–0.83.
Whatever separates the synthetic tag from the real tube, it is not how hard the confound is to
see. The revised working hypothesis is that fabrication tracks the **anatomical plausibility of
the confound category**: a synthetic tag is never mistakable for anatomy at any opacity — it is
either seen or not, with no ambiguous middle — whereas a real device sits inside a space of
plausible near-misses (other lines, catheters, vessels, fissures, rib edges) that a VLM's
medical-image priors can generate confident boilerplate about.

### 3.2 Experiment 2 — fabrication is grounded, not blind recitation

The competing explanation is that the VLM is not reading the image at all — a live concern,
since Lotfinia et al. (arXiv:2606.17710) show VLMs can answer chest-radiograph questions from
language priors without attending to the image. If V4b's descriptions were prior-emission
independent of pixels, the auditor's naming step would be meaningless even where it scores well.

We tested this with three per-crop conditions scored against each model's own unmodified-crop
baseline description: occlude the **relevant** region, occlude an equal-area **irrelevant**
region, or **swap** in a different same-kind crop. Verdict changes are broken down by
direction, because a combined "changed" rate conflates opposite outcomes: **pos->neg** (named
at baseline, not named after) is consistent with grounding; **neg->pos** (a *new* device named
after the edit) is fabrication merely relocated. All 12 cells are complete at n = 100.

| dataset | condition | model | pos->neg | neg->pos | unchanged | change rate |
|---|---|---|---|---|---|---|
| v4a | relevant | scout | 19 | 5 | 76 | 0.24 |
| v4a | relevant | pixtral | 25 | 22 | 53 | 0.47 |
| v4a | irrelevant | scout | 16 | 5 | 79 | 0.21 |
| v4a | irrelevant | pixtral | 27 | 24 | 49 | 0.51 |
| v4a | swap | scout | 8 | 8 | 84 | 0.16 |
| v4a | swap | pixtral | 27 | 23 | 50 | 0.50 |
| v4b | relevant | scout | 61 | 1 | 38 | 0.62 |
| v4b | relevant | pixtral | 66 | 5 | 29 | 0.71 |
| v4b | irrelevant | scout | 30 | 4 | 66 | 0.34 |
| v4b | irrelevant | pixtral | 13 | 2 | 85 | 0.15 |
| v4b | swap | scout | 20 | 21 | 59 | 0.41 |
| v4b | swap | pixtral | 17 | 17 | 66 | 0.34 |

**A measurement artifact requires that most of this table be read with care.** The flat gray
fill used to occlude the tag is *itself* routinely described by the VLM as "a gray rectangular
object... artificial and not part of the patient's anatomy... a label or marker" — language
that trips the exact `names_confound` rule the tag was designed to trigger (an object-word
combined with a brightness/artificiality word). Spot-checking the raw text confirms this
directly: **76 of 100** Scout descriptions under V4a *relevant* occlusion still flag positive,
matching the `unchanged` count, and reading them shows the model is describing the **occlusion
patch itself**, not residual tag content. The V4a *irrelevant* cell, whose superficially
similar 79/100 unchanged count might suggest the same problem, is a genuinely different
mechanism: an audit of that cell found 49 of 51 sampled descriptions explicitly mention
"white," i.e. they are correctly still detecting the untouched original tag, which irrelevant
occlusion never covers.

The consequence is specific and load-bearing. **V4a's `relevant` numbers understate true
grounding**, because a large share of its `unchanged` count is the occluder being misread as an
artificial marker rather than the model still perceiving a removed tag. **The V4a-vs-V4b
comparison is therefore invalid for this question**, and "V4a's verdict changed less than
V4b's" must not be read as "V4a is less grounded" — the measurement is biased toward
`unchanged` for V4a in a way it is not for V4b.

**The evidence that carries the conclusion is the V4b-internal contrast.** V4b's matcher
(`names_tube`) requires device-specific vocabulary — tube, drain, catheter, line, wire — which a
flat gray rectangle does not resemble the way it resembles a generic "artificial marker." Its
susceptibility to the artifact is empirically far lower: only **38 of 100** Scout descriptions
stay unchanged-positive under V4b *relevant* occlusion (against V4a's 76 of 100), and
spot-checking those found most describe a genuinely still-visible **second** device outside the
occluded region rather than a misread of the patch. Within V4b, both conditions use the **same
matcher on the same dataset**, differing only in *where* the occluder lands — so the
contamination cannot explain a difference between them:

- Occluding the **tube region** drops device-naming: pos->neg **61/100** (Scout), **66/100**
  (Pixtral).
- Occluding an equal-area **irrelevant** region drops it far less: **30/100** (Scout),
  **13/100** (Pixtral).

Device-naming tracks the tube pixels. **"V4b fabrication is pure image-independent
prior-emission" is not supported.**

The reconciled mechanism, read against the 50–83% control-hallucination measured on clean
crops: the VLM carries a strong **device-present prior** *and* grounds it on the salient region
— correctly reading a real tube when one is present, and **over-reading ambiguous normal
anatomy as a device** when none is. This is prior-driven over-reading of an ambiguous salient
region: neither blind recitation nor clean perception. It coheres with Experiment 1's finding
that plausibility, not visibility, is the operative axis — the tube fails because normal
anatomy affords a plausible device reading, and the tag never does.

Supporting but not load-bearing: V4b's swap condition is close to symmetric (pos->neg roughly
equals neg->pos — 20/21 Scout, 17/17 Pixtral), consistent with the model re-reading each new
image rather than pattern-locking onto a fixed answer. `text-identical-rate` was 0.0 in every
cell: the models never reproduce a baseline description verbatim, even under irrelevant
occlusion, so verbatim recitation is not the failure mode.

**Limitation for replication.** A texture-matched occluder (local mean intensity or a blurred
neighborhood) should replace the flat gray fill, so the occluder does not itself read as an
artificial rectangular object. This would clean the V4a arm and permit the cross-dataset
comparison that the present design cannot support.

### 3.3 Specialist VLM

The V4b mechanism above was established for general-purpose VLMs only, leaving one escape hatch:
maybe a radiology-fine-tuned VLM, carrying a device vocabulary the general models lack, would
clear the gate the general models failed. It does not — but the *way* it fails is the result.
Six additional models were run on the **identical** V4a/V4b crops and the same blind prompt
(byte-verified against the general arm's), scored with the existing project matchers. Three are
reportable; **three are excluded as harness output bugs, not model results**, and their zeros are
not findings: **phi3.5-vision** returned an empty string on all 200 crops in both arms;
**qwen2.5-vl-3b** produced the refusal "no image attached" on 164/200 V4a crops — the image never
reached the model; **MedGemma** returned 400 blank responses from a quantized-vision-tower bug,
diagnosed separately with a re-run pending. Reporting any of these as a naming failure would be a
fabrication of the opposite kind. *Three models reported; three excluded for harness output bugs.*

The three reportable models — qwen2.5-vl-7b, CheXagent-2-3B, and chexagent-8b — split cleanly on a
single axis: whether they **over-name** or **under-name** the device. Both failure modes clear
neither V4b gate, but for opposite reasons, and the split is the contribution.

**General VLMs over-name, and it is not a small-model artifact.** The V4b fabrication that
triggered the pivot was a general-VLM behavior: on tube-free controls, Llama-4-Scout fabricates a
device on **0.55** of clean crops (95% bootstrap CI [0.45, 0.65]) and Pixtral-12B on **0.83**
([0.75, 0.90]). qwen2.5-vl-7b — a *larger* general VLM, and the only new model with non-trivial
signal (V4a precision **0.63** / recall **0.33**) — over-names just as hard: its V4b
control-hallucination is **0.74**. Over-naming is therefore not an artifact of the 12B-and-under
general models that first exhibited it; it persists at 7B on a fresh model family, at a rate that
still fails the <0.10 control-hallucination bar sevenfold.

**The medical specialist under-names — a structural zero.** CheXagent-2-3B does not answer in free
text; it emits its own referring-expression grounding tokens
(`<|ref|> foreign object <|/ref|> <|box|>...<|/box|>`). Scored with a grounding-aware matcher that
holds it to the *same* genericness bar as the general models, it clears V4a's precision handsomely
(precision **0.9663**, recall **0.86**, control-hallucination **0.03**) — near the gate and far
below the fabrication bar. On V4b it scores exactly **0.0 / 0.0 / 0.0**. This is not a perception
failure: CheXagent produced a grounded label on **100/100** V4b tube-positive crops. It is a
*taxonomy* failure. Its entire observed grounding vocabulary is
`{foreign object, diaphragm, other lesion, artificial object, aortic enlargement, mediastinal}` —
**no device-specific class exists anywhere in its output space**: nothing that says tube,
catheter, line, drain, or wire, in either dataset. The specialist can localize the tube and cannot
*name* it, because "name it as a device" is not an option its taxonomy affords. Where the general
VLMs fail V4b by saying too much on clean crops, the specialist fails by having no word for the
thing it is grounding on.

**Under-naming survives a 3B->8B scale-up.** chexagent-8b, the larger sibling, answers in free-text
radiology-report prose rather than grounding tokens, and scores **0.0 / 0.0 / 0.0** on both arms:
its prose never reaches the device vocabulary the matchers key on, with no compensating control
false positives. The under-naming ceiling is thus not a 3B capacity limit — it clears a 3B->8B
scale-up unchanged — but a property of how the specialist's output space was designed. Naming a
device is a vocabulary-design decision made upstream of scale.

**One caution against an over-clean reading.** "Specialist under-names, general over-names" is the
axis, but it is not the *only* axis, and the general models are not uniformly safe wherever the
specialist is silent. Even on the easy **synthetic** V4a controls, fabrication is model-dependent:
Scout hallucinates on **0.02** of clean crops ([0.00, 0.05]) but Pixtral on **0.31**
([0.22, 0.40]) — a per-model fabrication *floor* that has nothing to do with domain
specialization. The clean "synthetic naming is safe" story is Scout-specific; another general VLM
fabricates on a third of the same easy controls. The auditor's reliability therefore turns on the
specific model on both axes at once — how much it over-names on clean crops *and* whether its
vocabulary can name the target at all — and neither axis is settled by the general-vs-specialist
label alone.

---

## 4. Results — supporting evidence and robustness

[CITATIONS — PENDING]

Five VLM-independent investigations probe the robustness of the results above. None is a
headline; each either strengthens a claim by an independent route or bounds an alternative
explanation.

### 4.1 The tube shortcut is severity-invariant

If the model were partly reading pathology and merely helped by the tube, reliance should
shrink on large, obvious pneumothoraces. It does not. Stratifying the 94 overlapping
pneumothorax-positive test images at the median bounding-box area (38,208.4 px²) and balancing
groups within each tier:

| tier | n (balanced group size) | acc. with tube | acc. without tube | reliance gap |
|---|---|---|---|---|
| low severity (area < median) | 20 | 1.0000 | 0.6000 | **0.4000** |
| high severity (area >= median) | 16 | 1.0000 | 0.5625 | **0.4375** |

Accuracy is perfect whenever a tube is present, at *any* lesion size, and collapses when it is
absent, at *any* lesion size. The gap is severity-invariant (40.0pp vs 43.75pp). The shortcut is
**device-keyed, not disease-burden-keyed** — the model is not using the tube as a hint about a
finding it also perceives; it is substantially substituting the tube for the finding.

### 4.2 The DFR result is not an artifact of an unturned knob

The mitigation numbers use single-draw DFR at C = 1.0. A canonical variant (regularization C
tuned by 5-fold cross-validation, weights averaged over 10 random balanced draws) was run to
check that the result is not a lucky hyperparameter:

| confound | metric | before DFR | standard DFR (C=1.0, 1 draw) | canonical DFR (C=0.001, 10 draws) |
|---|---|---|---|---|
| tube | ID accuracy | 0.9575 | 0.9650 | **0.9975** |
| tube | worst-group | 0.6700 | 0.8425 | **0.9275** |
| tube | reliance gap | 0.2875 | 0.1225 | **0.0700** |
| tag | ID accuracy | 0.9397 | 0.5937 | **0.6870** |
| tag | worst-group | 0.5000 | 0.5927 | **0.6813** |
| tag | reliance gap | 0.4397 | 0.0010 | **0.0057** |

On the real tube, the canonical variant strictly dominates on all three metrics: the reported
result is a *floor*, not a ceiling. On the synthetic tag, canonical DFR recovers some ID
accuracy (0.594 -> 0.687) while holding the gap near zero, but remains far below the 0.94
baseline — and we document the reason as an anomaly rather than a finding: the tag validation
set concatenates tagged and untagged copies of *every* image, leaving the logistic head unable
to separate the confound axis from the label axis in this symmetric setup, so it collapses
toward a coin flip regardless of C or draw count. The tag "closing" the gap is therefore partly
an artifact of that construction, and the tube result is the one we defend.

### 4.3 Site identity is redundantly encoded — a stronger restatement of the V6 null

V6's null zero-shot source-DFR result invites the reading that the model must instead be riding
a *site* signal. Probing the frozen V1 penultimate features (4,000 NIH test images vs 7,572
PadChest images), a hospital detector separates the two sites at AUC **0.9993** — site identity
is almost perfectly encoded. But the falsifying control refutes the inference: projecting that
site direction out of the PadChest features and re-scoring the **frozen disease head** leaves
disease AUC **unchanged at 0.6436 -> 0.6436 (Δ 0.0000)**. Site is massively present in the
representation, and the disease head does not read it.

A second control bounds how much that projection even accomplished: a fresh probe on the
*projected* features still reaches AUC **0.9990**, a drop of only **0.0003**. A rank-1
projection removes essentially none of the site signal, which instantly re-emerges from the
remaining dimensions of the 1024-d space. This corrects an earlier over-claim in the project's
own record: the correct statement is that site information is redundantly encoded across many
directions, and seriously ablating it would require full subspace projection or adversarial
training rather than a single vector removal. The disease-AUC null (Δ 0.000) stands on its own
regardless, since it is measured on the frozen head.

### 4.4 Label-free OOD attribution: the synthetic-vs-real asymmetry

Generalizing the zero-shot source-DFR test from the real tube to the synthetic tag sharpens the
picture. For the tag-trained model, NIH reference AUC is **0.7407** and PadChest AUC before
intervention is **0.6345** (OOD drop **0.1062**). After tag source-DFR on NIH only, with zero
PadChest labels: PadChest AUC **0.5520**, a change of **−0.0825**.

Source-side mitigation of the tag does not merely fail to transfer — it **actively hurts**
transfer. Set beside the tube's null (**+0.005**), the asymmetry is informative:

| confound | zero-shot source-DFR change on PadChest |
|---|---|
| tube (real) | **+0.005** (null) |
| tag (synthetic) | **−0.083** (harmful) |

This is the same asymmetry the linear probe (0.690) and the correlation sweep report from
another direction, and it is the key caveat on the whole synthetic arm: a *pure* synthetic
shortcut at p = 0.95 is a near-total representation override. The tube model retains genuine
disease signal alongside its shortcut; the tag model at high *p* largely stops learning the
disease, so removing the tag axis from the head leaves too little behind — in-distribution
(0.94 -> ~0.69) and out-of-distribution (0.634 -> 0.552) alike. Synthetic confounds prove the
*method*; they systematically overstate the *severity* of the phenomenon relative to real ones,
and results measured on them should not be extrapolated to real shortcuts without the real-arm
check.

### 4.5 Where reliance turns on: the correlation sweep

Reliance is not a binary property of "having a confound." Sweeping correlation strength *p*
(single training seed per point):

| p | reliance gap | worst-group before -> after DFR | ID before -> after DFR |
|---|---|---|---|
| 0.60 | 0.000 | 0.750 -> 0.653 | 0.750 -> 0.660 |
| 0.70 | 0.010 | 0.745 -> 0.670 | 0.755 -> 0.668 |
| 0.80 | 0.075 | 0.722 -> 0.672 | 0.796 -> 0.639 |
| 0.90 | 0.294 | 0.585 -> 0.605 | 0.879 -> 0.618 |
| 0.95 | 0.440 | 0.500 -> 0.593 | 0.940 -> 0.617 |

The gap rises **monotonically** with *p*, and there is an **onset threshold**: below p ≈ 0.7 the
correlation is too weak to exploit and the model ignores the tag and learns real signal (test
AUC ≈ 0.83, matching clean V1). DFR's in-distribution cost is correspondingly
**strength-dependent, not binary** — negligible where there is no shortcut to remove, large
where the shortcut was allowed to dominate. The DFR-recovery step was re-run at 3 seeds on the
frozen per-*p* checkpoints (varying the balanced sampling and head fit only, not training):
every single-seed value above falls within roughly one standard deviation of those means (e.g.
at p = 0.95, 0.593 vs 0.609 ± 0.022), so the sweep's conclusions do not hinge on the seed. The
*training*-seed variance of the reliance gaps themselves is not claimed.

### 4.6 A fifth investigation, reported as blocked

Severity-**aware** mitigation — testing whether mitigation can be targeted by lesion burden —
was **not run**. Severity labels exist only for the 94-image test split, and no proxy was
fabricated to fill the gap. It is reported here as blocked rather than omitted.

---

## 5. Related work

[CITATIONS — PENDING]

**Shortcut learning in chest radiography** is well established. Zech et al. showed CNNs
generalize worse across hospitals than within them, and can identify the hospital system with
~99.95–99.98% accuracy. DeGrave, Janizek & Lee showed COVID-19 CXR models keyed on text markers
and patient positioning rather than pathology. Jiménez-Sánchez et al. present a manual and
statistical shortcut-detection case study on chest X-rays. These works establish that shortcuts
exist and are dangerous, and diagnose them by expert analysis or statistical association; none
automates the *naming* step through a general-purpose model.

**Correcting shortcuts generatively.** RoentMod (arXiv:2509.08640) pairs a medical image
generator with an image-to-image editor to produce counterfactual CXRs with specified findings,
reporting discrimination improvements of 3–19% AUC internally and 1–11% externally, with 93% of
generated images rated realistic by readers. It is the closest correction-side prior art and the
method we most directly contrast against: it relies on a generative diffusion pipeline and
fine-tuning, where our naming step uses one free API pass over ~200 crops with no generative
model at all. That is a genuine trade rather than a strict improvement — RoentMod's
correction-only design never had to clear the bar our naming step failed.

**Mitigation baselines.** GroupDRO optimizes worst-group loss but requires group labels; JTT
upweights an ERM model's errors to approximate group robustness without them; DFR retrains only
the last layer on group-balanced data, reaching 92.9% worst-group on Waterbirds and 70.1% on
CivilComments at a fraction of GroupDRO's cost, on the premise that the backbone already encodes
core features. Follow-up work (arXiv:2308.00473) questions whether last-layer retraining always
suffices — a caution our synthetic-tag results directly corroborate, since on a *pure*-shortcut
backbone DFR's in-distribution recovery is incomplete. These methods assume the groups are
already known; our loop attempts to discover and name the confound first.

**Concurrent work on the same shortcut.** Zhao et al. (arXiv:2604.17110) use an autonomous LLM
coding-agent pipeline that, among other tasks, reduces chest-drain reliance for pneumothorax
classification on NIH-derived data, reporting 60%->31% and 50%->18% reliance reduction on two
datasets. Same target shortcut, different mechanism (agentic pipeline search vs. last-layer
retraining), different metric (% reliance vs. worst-group accuracy and reliance gap). It reports
no VLM-naming step, no control-hallucination measurement, and no cross-site transfer test. We
read it as independent, contemporaneous confirmation that this real shortcut is a live target
reached by a different route — corroborating, not competing.

**VLM-based shortcut auditing outside medicine.** SpurLens (arXiv:2503.08884) uses GPT-4 plus
open-set detectors to discover spurious visual cues multimodal LLMs over-rely on in natural
images, and separately measures hallucination amplification (objects hallucinated up to 10x more
often when a spurious cue is present). RaVL (arXiv:2411.04097) discovers region-level spurious
correlations inside a fine-tuned VLM's own errors and mitigates with a region-aware loss (+8.2%
worst-group accuracy; 191% discovery improvement over prior methods, across 654 VLMs). Both are
close mechanism analogues, but neither touches medical imaging, blind clinical naming, or a
control-hallucination-gated metric. MIMM-X (arXiv:2511.22990) disentangles a **pre-specified**
confounder via mutual-information objectives — the opposite of a blind design, and evidence that
the standard alternative still assumes foreknowledge of the confound.

**VLM reliability on chest radiographs** bears directly on our V4b result. HalluCXR
(arXiv:2605.20469) benchmarks 6 VLMs on 856 chest radiographs, finding 61.9–82.3% overall and
36.4–80.2% "clinically dangerous" hallucination rates, with severity highest on normal images
(51.9% severity-3 on normals). This corroborates our 0.50–0.83 control-hallucination on clean
crops in a different model population on a different task (diagnostic findings vs. device
naming). Lotfinia et al. (arXiv:2606.17710) show VLMs can answer CXR questions from language
priors without attending to the image — the direct threat to our naming step's interpretation,
which our Experiment 2 addresses and does not support as an explanation of V4b (§3.2).
Specialist CXR-VLMs (CXR-LLaVA, arXiv:2310.18341; LLaVA-Rad) are fine-tuned on radiology data;
we deliberately used a general-purpose, non-fine-tuned model to test the lower-barrier claim.

**Ground-truth dependency.** NEATX (arXiv:2309.02244) supplies the non-expert chest-drain
annotations our real-confound experiments rely on. This is a data dependency, credited
distinctly from prior art.

### 5.1 Positioning

| method | data modality | uses report corpus? | control-hallucination gate? | models tested |
|---|---|---|---|---|
| **This work (auditor)** | chest X-ray (NIH, PadChest); NEATX device labels | **No** — blind prompt on saliency crops only | **Yes** — separate gate at <10% on confound-free controls; the gate that failed V4b | 2 general-purpose VLMs: Llama-4-Scout, Pixtral-12B `[EXP3 RESULTS — PENDING: specialist arm]` |
| **LADDER** (Ghosh et al., Findings of ACL 2025) | image → **caption** (BLIP/GPT-4o) + activation-to-text | **Yes** — captioner required (caption-free variant only in an ablation) | **No** — worst-group accuracy + slice Precision@10 | reasoning LLMs (GPT-4o, Claude, Gemini, LLaMA); classifiers on NIH-CXR, RSNA, VinDr, Waterbirds, CelebA, MetaShift |
| **HalluCXR** (arXiv:2605.20469) | chest X-ray (856 radiographs) | Not reported in our sources | No — it *measures* hallucination as the object of study (61.9–82.3% overall; severity highest on normals), rather than gating a capability on it | 6 VLMs (identities not in our sources) |
| **arXiv:2604.17110** (Zhao et al.) | chest X-ray (NIH-derived) | Not reported in our sources | **No** — reports % reliance reduction (60%->31%, 50%->18%); no naming step to gate | LLM coding-agent pipeline; no VLM-naming arm reported |

The nearest prior art, **LADDER**, shares our premise — name a classifier's shortcut with a
foundation model, in radiology, without attribute labels — but reads the image as *text* (a
caption plus projected activations) and evaluates discovery only through the downstream accuracy
of the mitigation it enables. Our method reads the classifier's saliency region as *pixels*,
through a blind-prompted general VLM, and evaluates the name directly against ground truth under
a control-hallucination gate. That gate is not a detail: it is what converts "the VLM produced a
plausible shortcut description" into a testable claim, and it is why we can report — rather than
absorb into a mitigation score — that blind naming *fails* on a real device confound (V4b) while
succeeding on a synthetic one (V4a).

The column that separates this work from every other row is **control-hallucination as a gate**.
LADDER evaluates via worst-group accuracy; HalluCXR measures hallucination as its subject; Zhao
et al. have no naming step to gate; our design makes a fabrication rate on clean images a
*pass/fail condition on the capability itself*, which is why V4b is reported as a failure rather
than as a 0.57-precision result. (Two cells our own source documents do not state either way —
the "uses report corpus?" status for HalluCXR and Zhao et al., and the identity of HalluCXR's six
models — are left marked accordingly and should be filled from the primary sources.)

---

## 6. Limitations

[CITATIONS — PENDING]

**Single pathology, single primary corpus.** Every result concerns pneumothorax on NIH
ChestX-ray14 and its cross-site evaluation on PadChest. The mechanism proposed in §3 —
fabrication driven by the anatomical plausibility of the confound category — predicts that other
device-adjacent pathologies would behave similarly, but this is a prediction, not a measurement.

**Non-expert ground truth for the real confound.** NEATX's chest-drain annotations come from two
data-science students, not radiologists, with "moderate to almost perfect" agreement against
expert labels reported in the source paper. This is adequate for inducing a training-time
enrichment signal and defining a mitigation target, but it is a noisier ground truth than a
radiologist-drawn segmentation — and it bears most heavily on exactly the number the paper turns
on: V4b's precision and recall are scored *against these labels*. A radiologist-grade corpus
(e.g. CANDID-PTX, whose access requires an ethics-training pathway we did not complete in time)
would be the appropriate ground truth for re-scoring the naming gate specifically.

**Two-site validation.** Two sites are two points, not a trend. The cross-site drop (0.192) is
real and Zech-magnitude, and the §4.3 controls rule out two candidate mechanisms — a disease head
riding a decodable site signal (Δ 0.000), and label-free covariate shift correctable by standard
test-time adaptation (BN-adaptation −0.004; TENT −0.122, actively harmful and collapsing toward
chance on this single-logit imbalanced head). But most of the gap remains genuine, unattributed
distribution shift. Candidate causes this design cannot separate include scanner and acquisition
differences, population differences, the two datasets' different label-generation pipelines (NIH's
NLP-only labeler vs. PadChest's NLP-plus-manual process), residual intensity artifacts despite the
8-bit fix, and the composition of the 573-positive / 7,000-negative working subset. Label
semantics are known to differ in a direction that can inflate the measured gap — among PadChest's
573 pneumothorax-positive cases, only **15.7%** carry pneumothorax as the *sole* finding, and the
most common co-finding is a chest drain tube (**109 cases, ≈19%**) — but the working-subset CSV
lacks the projection and method fields needed to quantify this, so it is bounded qualitatively
rather than estimated. A third site would turn two points into a trend; the design accommodates
one without rework, but we did not run it.

**Single-GPU scale.** All training ran on one consumer GPU (RTX 3050, 6GB), which shaped batch
size (16, after an out-of-memory failure at 32), resolution (224px, standard for this
literature), and the decision to work with subsets for several experiments — most visibly the
7,572-image PadChest working set against the full release's 160,861. The correlation sweep
carries one *training* seed per point for the same reason; its trend (monotone reliance growth)
is clear, but individual point values could shift under a different seed. The 3-seed baseline
table is the result we defend most strongly on variance grounds. Nothing measured here appears
to depend on the hardware in a way that would not reproduce at scale — but larger-scale
replication was out of reach and is not claimed.

**The difficulty boundary is bounded on one axis, unmapped on the others.** Experiment 1 rules
out *visibility* as the axis along which naming degrades into fabrication, and the multi-VLM
replication closes the "a different model would fix it" escape hatch for general-purpose VLMs
(55% and 83% fabrication on clean crops, two independent families). What remains open is where
between an unambiguous synthetic marker and a subtle real device the boundary actually sits along
the *plausibility* axis the mechanism implicates — and whether prompt design, crop resolution, or
abstention/confidence mechanisms shift it. We deliberately did not attempt to rescue V4b's
precision and recall through prompt engineering, because the pre-registered design specified that
a V4b failure after a V4a pass triggers a pivot rather than iterative tuning. That is a
principled stopping rule, and it is also the reason the boundary is characterized rather than
mapped. `[EXP3 RESULTS — PENDING]` — the specialist-VLM arm addresses the narrower question of
whether radiology fine-tuning moves the boundary at all.

**Interpretive claims corrected mid-project.** Two claims were written before the control that
tests them existed, and both were withdrawn once it was run: "the pure-shortcut backbone is
hollow" (falsified by the linear probe: AUC 0.690, above chance 0.50) and "mitigation validated
across three settings" (falsified by the zero-shot source-DFR null: +0.005, after the +0.082
"recovery" was correctly re-identified as supervised target adaptation). We report both, because
the failure mode they share — asserting an interpretation the experiment does not license — is
the same one this paper's method is designed to catch in a classifier, and there is no principled
reason to exempt the authors from it.
