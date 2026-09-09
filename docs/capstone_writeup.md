# Auditing Chest X-ray Classifiers for Spurious Shortcuts: Where Blind VLM Naming Succeeds, Where It Fabricates, and What Survives the Failure

*Unified capstone writeup. Assembled from project components (CO1–CO4 deliverables, Phase 5 extension experiments, CO3 spike-by-spike results), with connective narrative. Every quantitative claim traces to measured results recorded in PROGRESS.md or experimental deliverables.*

---

## Abstract

Deep chest X-ray classifiers routinely reach high in-distribution accuracy while relying on spurious shortcuts — features correlated with the label in training but not causally tied to the pathology — and fail silently when that correlation breaks at a new site. We build an *auditor* that runs a four-stage loop over a pneumothorax classifier: **detect** that accuracy depends on a confound, **localize** the confound by attribution, **name** it in clinical language using a general-purpose vision-language model (VLM) that is never told what to look for, and **mitigate** the reliance by last-layer retraining. The study is executed as six validation-gated spikes, each with a pre-registered pass/fail threshold and a stop-and-report point.

Detection and localization transfer cleanly from a synthetic confound to a real device confound: we induce a measurable reliance gap (44 percentage points synthetic; 28.75 points on real chest tubes) and localize it (Grad-CAM peak-in-region 100% synthetic; side-consistency 96.6% real). Blind VLM naming, however, does not transfer. It passes at near-ceiling on the synthetic marker (precision 0.978, recall 0.90, control-hallucination 0.02) but **fails on the real chest tube by fabrication**: on tube-free control crops the VLM invents devices — endotracheal tubes, central lines, catheters — at a rate of 0.55 (Llama-4-Scout) and 0.83 (Pixtral-12B) in the multi-VLM replication. This cross-model fabrication is the honest negative result at the center of the paper: an auditor built on off-the-shelf general VLMs would raise a fabricated shortcut alarm on a majority of clean images, which is exactly the false-alarm failure the design's precision-over-recall gate was built to catch. What survives the naming failure is mitigation, which never depended on the VLM: Deep Feature Reweighting moves worst-group (pneumothorax-without-tube) accuracy from 0.67 to 0.79 as a 3-seed mean while preserving in-distribution accuracy, using no target-distribution labels.

Finally, a real US-to-Spain cross-site AUC drop of 0.192 (95% CI [0.159, 0.225], bootstrap) exists, but a zero-shot, label-free test of whether reducing the known source shortcut narrows it returns a point estimate indistinguishable from zero (+0.005, 95% CI [−0.018, +0.028]) — underpowered to rule out a small-to-moderate effect but not evidence of a real recovery either; the drop is distribution shift this design cannot attribute to the shortcut. We report the naming failure as a headline rather than a footnote, and we report two interpretive claims we withdrew after running their falsifying controls.

---

## 1. Introduction

Deep chest X-ray (CXR) classifiers can reach high in-distribution accuracy while keying on features that are correlated with the label in the training data but not causally related to the pathology — scanner text, laterality and positioning markers, hospital- or device-specific artifacts. Zech et al. showed that CNNs generalize worse across hospitals than within them and can identify the source hospital system with ~99.95–99.98% accuracy; DeGrave, Janizek & Lee showed COVID-19 CXR models keying on text markers and patient positioning rather than pathology. Such models appear accurate and fail under distribution shift, which is a patient-safety problem: a model that keys on the chest tube used to *treat* a pneumothorax rather than on the pneumothorax itself will silently mispredict exactly on the untreated patients where correct detection matters most.

Auditing a classifier for this failure has four parts: (1) **detect** that it relies on a shortcut, (2) **name** the shortcut in clinical terms without being told what to look for, (3) **mitigate** the reliance, and (4) **prove recovery** under out-of-distribution shift. Prior work tends to address only some. Shortcut-diagnosis papers (Zech; DeGrave; Jiménez-Sánchez et al.) detect shortcuts by expert or statistical analysis but do not automate the naming step. Generative-correction methods such as RoentMod edit images with a diffusion pipeline to correct shortcuts, at the cost of a heavy generative apparatus and fine-tuning. Worst-group robustness methods (GroupDRO, JTT, DFR) mitigate but assume the groups are already known. The nearest label-free naming prior art, LADDER, names a classifier's shortcut with a foundation model in radiology, but routes the image through a captioner plus a text-only LLM and evaluates discovery only through downstream mitigation accuracy — with no false-alarm gate.

**The gap this paper targets** is prompt-free, generative-model-free naming, wired into a full detect→localize→name→mitigate loop and scored directly against ground truth under a *control-hallucination gate*: the rate at which the auditor names a device on a confound-free image. That gate is the design's identity. For an auditor, a fabricated confound costs more than a missed one, because a tool that invents shortcuts is worse than no tool; so precision is gated above recall, and the fabrication rate on clean images is a pass/fail condition on the capability itself, measured separately.

**Contributions.**
- A validation-gated auditor loop that detects, localizes, and mitigates a real device shortcut in a pneumothorax classifier with no target-distribution labels, built on a confound-agnostic core that serves synthetic, real-device, and cross-site settings from one implementation.
- The paper's central finding, reported honestly as a negative result: blind, general-purpose VLM naming **succeeds on a synthetic confound and fails on a real device confound by fabrication**, and the failure is a cross-model property (Llama-4-Scout and Pixtral-12B both fabricate devices on a majority of clean crops), independently corroborated by HalluCXR.
- Mechanism experiments that bound *why* naming breaks: fabrication is not driven by the confound's visibility (a recall cliff, not a hallucination cliff), and it is grounded in the image rather than blind recitation — it is prior-driven over-reading of an ambiguous salient region.
- A zero-shot, label-free source-side test of whether a known shortcut explains a measured cross-site generalization gap, reporting a null result: the 0.192 US-to-Spain drop is distribution shift not shown to be shortcut-driven.
- An account of two interpretive claims withdrawn after running the specific controls that falsified them, treated as a first-class result rather than hidden.

---

## 2. Related Work

**Shortcut learning in chest radiography** is well established. Zech et al. showed CNNs generalize worse across hospitals than within them, and can identify the hospital system with ~99.95–99.98% accuracy. DeGrave, Janizek & Lee showed COVID-19 CXR models keyed on text markers and patient positioning rather than pathology. Jiménez-Sánchez, Juodelyte, Chamberlain & Cheplygina present a manual and statistical shortcut-detection case study on chest X-rays. These works establish that shortcuts exist and are dangerous, and diagnose them by expert analysis or statistical association; none automates the *naming* step through a general-purpose model.

**Correcting shortcuts generatively.** RoentMod (arXiv:2509.08640) pairs a medical image generator (RoentGen) with an image-to-image editor to produce counterfactual CXRs with specified findings while preserving anatomy, reporting discrimination improvements of 3–19% AUC internally and 1–11% externally, with 93% of generated images rated realistic by readers. It is the closest correction-side prior art and the method we most directly contrast against: it relies on a generative diffusion pipeline and fine-tuning, where our naming step uses one free API pass over ~200 crops with no generative model at all. That is a genuine trade rather than a strict improvement — RoentMod's correction-only design never had to clear the bar our naming step failed.

**Mitigation baselines.** GroupDRO (Sagawa et al., 2020) optimizes worst-group loss but requires group labels; JTT (Liu et al., 2021) upweights an ERM model's errors to approximate group robustness without them; DFR (Kirichenko et al., 2022) retrains only the last layer on group-balanced data, reaching 92.9% worst-group on Waterbirds and 70.1% on CivilComments at a fraction of GroupDRO's cost, on the premise that the backbone already encodes core features. Follow-up work (arXiv:2308.00473) questions whether last-layer retraining always suffices — a caution our synthetic-tag results directly corroborate, since on a *pure*-shortcut backbone DFR's in-distribution recovery is incomplete. Hameed, Nanfack & Belilovsky (arXiv:2409.14637) respond to the same limitation by retraining on features pooled across all layers rather than only the last, reporting worst-group gains over standard DFR on Waterbirds-style benchmarks; we use standard last-layer DFR throughout, so an all-layer variant is a direct, cheap extension to try against our synthetic-tag collapse case (§6.4.2) rather than a competing claim. These methods assume the groups are already known; our loop attempts to discover and name the confound first.

**Concurrent work on the same shortcut.** Zhao et al. (arXiv:2604.17110) use an autonomous LLM coding-agent pipeline that, among other tasks, reduces chest-drain reliance for pneumothorax classification on NIH-derived data, reporting 60%→31% and 50%→18% reliance reduction on two datasets. Same target shortcut, different mechanism (agentic pipeline search vs. last-layer retraining), different metric (% reliance vs. worst-group accuracy and reliance gap). It reports no VLM-naming step, no control-hallucination measurement, and no cross-site transfer test. We read it as independent, contemporaneous confirmation that this real shortcut is a live target reached by a different route — corroborating, not competing.

**Foundation-model shortcut auditing (closest analogues to the naming step).** LADDER (Ghosh et al., Findings of ACL 2025) is the nearest medical-imaging prior art: it uses an LLM's latent domain knowledge to hypothesize a classifier's biases without pre-specified attribute labels, and evaluates partly on chest X-rays. Three differences make it a neighbor rather than the same method: (i) it reads the image as *text* (BLIP/GPT-4o captions plus activation-to-text projection), reasoning with a text-only LLM, where we feed the Grad-CAM crop **pixels** to a general VLM with no captioner and no report corpus; (ii) it scores discovery via slice Precision@10 and downstream worst-group accuracy, with **no control-hallucination gate**; (iii) its discovery is in service of mitigation, so a naming failure like our V4b would be absorbed into a mitigation score rather than surfaced. SpurLens (arXiv:2503.08884) uses GPT-4 plus open-set detectors to discover spurious cues multimodal LLMs over-rely on in natural images, and measures hallucination amplification (objects hallucinated up to 10× more often when a spurious cue is present). RaVL (arXiv:2411.04097) discovers region-level spurious correlations inside a fine-tuned VLM's own errors and mitigates with a region-aware loss (+8.2% worst-group accuracy; 191% discovery improvement, across 654 VLMs). Both are close mechanism analogues but neither touches medical imaging, blind clinical naming, or a control-hallucination-gated metric. MIMM-X (arXiv:2511.22990) disentangles a **pre-specified** confounder via mutual-information objectives — the opposite of a blind design, and evidence that the standard alternative still assumes foreknowledge of the confound. Grad-CAM-style attribution (Selvaraju et al.) localizes *where* a model looks — our localization step uses it — but does not name what it found.

**VLM reliability on chest radiographs** bears directly on our naming result. HalluCXR (arXiv:2605.20469) benchmarks 6 VLMs on 856 chest radiographs, finding 61.9–82.3% overall and 36.4–80.2% "clinically dangerous" hallucination rates, with severity highest on normal images (51.9% severity-3 on normals). This corroborates our 0.50–0.83 control-hallucination on clean crops in a different model population on a different task (diagnostic findings vs. device naming). Lotfinia et al. (arXiv:2606.17710) show VLMs can answer CXR questions from language priors without attending to the image — the direct threat to our naming step's interpretation, which our Experiment 2 addresses and does not support as an explanation of the failure (§6.2). Specialist CXR-VLMs (CXR-LLaVA, arXiv:2310.18341; LLaVA-Rad) are fine-tuned on radiology data; we deliberately used a general-purpose, non-fine-tuned model to test the lower-barrier claim.

**Ground-truth dependency.** NEATX (Cheplygina et al., arXiv:2309.02244) supplies the non-expert chest-drain annotations our real-confound experiments rely on. This is a data dependency, credited distinctly from prior art. Sourget et al. (arXiv:2507.21291) independently draw on NEATX alongside NIH-CXR14 and MIMIC-CXR to audit fairness and robustness of CLIP-based chest-X-ray models across demographic subgroups — a companion use of the same annotation resource for a different auditing question (demographic fairness rather than device-confound naming) that corroborates NEATX's suitability as a shared ground-truth source. Rafferty & Rajan (arXiv:2509.15107) catalog label-quality, domain-shift, and bias limitations across the public chest-radiography datasets this paper itself depends on (NIH ChestX-ray14, PadChest); their findings on inter-dataset label-pipeline inconsistency are the general case of the specific label-generation-pipeline difference we flag as an unattributed candidate cause of the V6 cross-site gap (§8).

### 2.1 Positioning

| method | data modality | uses report corpus? | control-hallucination gate? | models tested |
|---|---|---|---|---|
| **This work (auditor)** | chest X-ray (NIH, PadChest); NEATX device labels | **No** — blind prompt on saliency crops only | **Yes** — separate gate at <10% on confound-free controls; the gate that failed V4b | 2 general-purpose (Llama-4-Scout, Pixtral-12B) + specialist/scale arm (CheXagent-2-3B & -8B, qwen2.5-vl-7b) — over-name vs under-name axis |
| **LADDER** (Ghosh et al., Findings of ACL 2025) | image → **caption** (BLIP/GPT-4o) + activation-to-text | **Yes** — captioner required | **No** — worst-group accuracy + slice Precision@10 | reasoning LLMs (GPT-4o, Claude, Gemini, LLaMA); classifiers on NIH-CXR, RSNA, VinDr, Waterbirds, CelebA, MetaShift |
| **HalluCXR** (arXiv:2605.20469) | chest X-ray (856 radiographs) | **Yes** — radiologist reports supplement CheXpert labels as ground truth | No — it *measures* hallucination as the object of study (61.9–82.3% overall; severity highest on normals) | 6 VLMs: BiomedCLIP, Gemini 3 Flash, GPT-5.1, InternVL2, LLaVA-NeXT-Mistral, Qwen2-VL |
| **Zhao et al.** (arXiv:2604.17110) | chest X-ray (NIH-derived) | **No** — clinician plain-language description, not report text | **No** — reports % reliance reduction (60%→31%, 50%→18%); no naming step to gate | LLM coding-agent pipeline; no VLM-naming arm |

The nearest prior art, LADDER, shares our premise — name a classifier's shortcut with a foundation model, in radiology, without attribute labels — but reads the image as text and evaluates discovery only through the downstream accuracy of the mitigation it enables. Our method reads the classifier's saliency region as pixels, through a blind-prompted general VLM, and evaluates the name directly against ground truth under a control-hallucination gate. That gate is what converts "the VLM produced a plausible shortcut description" into a testable claim, and it is why we can report — rather than absorb into a mitigation score — that blind naming *fails* on a real device confound while succeeding on a synthetic one. What appears genuinely novel after an exhaustive July-2026 search: no paper (a) performs blind general-VLM naming of a CXR shortcut scored on precision/recall/control-hallucination, or (b) runs a zero-shot, label-free source-side intervention to test whether a known shortcut explains a cross-site gap, reporting the null honestly.

---

## 3. Methods

### 3.1 Overview

We audit a chest X-ray classifier for reliance on spurious shortcuts through a four-stage loop: **detect** that the classifier's accuracy depends on a confound, **localize** the confound via attribution, **name** it in clinical language using a general-purpose vision-language model (VLM) that is never told what to look for, and **mitigate** the reliance by last-layer retraining. The loop is evaluated on a synthetic confound, a real device confound, and a real second site. The study was executed as six validation-gated spikes (V1–V6), each with a pre-registered pass/fail threshold and a stop-and-report point, so that a failed stage terminates a claim rather than being iterated around.

Architecturally, this discipline produced a **confound-agnostic core**: the attribution (Grad-CAM) and mitigation (DFR) machinery take a generic confound description — a region box, or a `(confound, label)` column pair — so the same modules serve the synthetic tag, the real chest tube, and the cross-site case without modification. The one component with an external, rate-limited dependency — the VLM-naming step — is isolated behind a `crops/ + manifest.json → predictions.jsonl` file contract, so the provider is interchangeable and no downstream computation depends on it. No method in the mitigation path imports the naming module, enforcing by construction that the mitigation result cannot depend on the VLM.

### 3.2 Data and model

The primary corpus is NIH ChestX-ray14 (112,120 images, resized to 224×224), with patient-disjoint train/validation/test splits (73k/13k/25.6k) built from the official split lists; image-level splitting is never used, as it leaks a patient's other studies across splits. The target pathology is pneumothorax (roughly 3–5% prevalence depending on split). The classifier is an ImageNet-pretrained DenseNet-121 trained with automatic mixed precision, batch size 16, and a `pos_weight`-corrected binary cross-entropy loss for the class imbalance.

Real chest-drain ground truth comes from **NEATX** (Cheplygina et al., arXiv:2309.02244), which supplies non-expert chest-drain annotations for a subset of ChestX-ray14 and PadChest. The second site is **PadChest** (Hospital San Juan, Spain), used in its 224×224 release; a 7,572-image working subset (573 pneumothorax-positive, 6,999 negative) was extracted under a disk constraint. The physical extraction from the 12.89GB release archive was a one-time step outside the package (no script for it is checked in); within the resulting 7,572-image CSV index, row selection when further capping is needed is a uniform random sample (`build_padchest_index`'s `df.sample(random_state=seed)`), not stratified by label. This working subset is then split **patient-disjointly** into a 2,283-image adaptation set (`dfr` split, used only by the supervised-target-adaptation arm to fit a last layer) and a 5,289-image held-out **test** set, used for the baseline PadChest AUC (0.650) and the bootstrap CI reported in §5.3. The zero-shot source-DFR arm fits its mitigation entirely on NIH data — it never touches a PadChest image for fitting — so it has no leakage risk to guard against and evaluates on the full 7,572-image set instead, giving a very slightly different baseline (0.6505 vs. 0.650) than the held-out split; that difference is sampling noise between a set and its own 70%-larger superset, not a discrepancy. All three V6 arms (baseline, supervised adaptation, zero-shot source-DFR) are evaluated in a manner consistent with the PadChest data each one is permitted to touch.

### 3.3 V1 — pipeline sanity

A baseline pneumothorax classifier is trained end-to-end and gated on in-distribution test AUC ≥ 0.75. This spike exists solely to establish that any subsequent degradation is attributable to an intervention rather than to a broken pipeline.

### 3.4 V2 — shortcut induction

A synthetic confound — a 20×20 opaque corner tag — is stamped onto training images with correlation strength *p* to the pneumothorax label. Reliance is quantified as the **reliance gap**: accuracy with the confound aligned as trained, minus accuracy with the confound removed, measured on a class-balanced subset. A third condition (tag *flipped*, i.e. anti-correlated) provides a lower bound. The gate is a gap > 20 percentage points. A companion **correlation-strength sweep** trains the same architecture at *p* in {0.6, 0.7, 0.8, 0.9, 0.95} to characterize how reliance and its mitigability scale with confound strength.

### 3.5 V3 — attribution localizes the confound

Grad-CAM (`pytorch_grad_cam.GradCAM`, on `features.norm5` — DenseNet-121's final dense-block output — targeting the model's single positive logit) is computed on the shortcut-reliant checkpoint. Localization is scored with a pointing game (`peak_in_region`) against a generic box `(y0, x0, y1, x1)`, plus a `mass_fraction` statistic. Crucially, the same images with the confound *removed* serve as a control, which rules out "that corner is always salient" as an alternative explanation. All attribution geometry is confound-agnostic: no function knows whether the box contains a synthetic tag or a real device, so the identical code serves the synthetic and real settings.

For the real chest tube, `peak_in_region` is not usable: NEATX supplies only image-level tube presence plus a coarse left/right/both **side** label, never a pixel bounding box, so no precise ground-truth box exists to point into. The tube's localization metric is instead `peak_on_side` (a left/right pointing-game check, respecting the radiological convention that patient-left appears on the image's right) — a coarser metric than the synthetic tag's, dictated by the annotation resolution NEATX provides, not chosen to produce a more favorable number.

### 3.6 V4 — blind VLM naming

Salient crops from confound-positive images, plus confound-free control crops, are exported with a manifest carrying exact `has_confound` labels the VLM never sees. Crops are captioned by a general-purpose, non-fine-tuned VLM under a **blind prompt** that never mentions the target ("describe anything artificial, non-anatomical, or out of place... if normal, say so"). Free-text descriptions are scored by conservative keyword matchers — `names_confound` for the synthetic-marker vocabulary, `names_tube` for device-specific vocabulary (tube / drain / catheter / line / wire) — into precision, recall, and **control-hallucination**: the rate at which a device or marker is named on a confound-free crop. Matchers are unit-tested not to fire on generic anatomy language.

The naming stage is deliberately split into two gates with different bars, because they test different things:

- **V4a (synthetic tag)** proves the *method* works where ground truth is exact and the confound is unambiguous. Gate: precision ≥ 0.90, recall ≥ 0.90, control-hallucination < 10%.
- **V4b (real chest tube)** proves the method *transfers*. Gate: precision ≥ 0.80, recall ≥ 0.70, control-hallucination < 10%.

Precision is gated above recall by design: for an auditor, a fabricated confound costs more than a missed one, since a tool that invents shortcuts is worse than no tool. The control-hallucination bar is measured separately and is the hardest to relax. The original design specified a six-model roster (Llama-4-Scout, Llama-4-Maverick, Gemini, GPT-4o, Pixtral-12B, and the specialist arm reported separately in §6.3); two of those six did not reach a result before the run: Maverick was retired by its provider (Groq) before it could be run, and GPT-4o was excluded for lack of a provisioned API token — both practical access constraints on a zero-budget project, not results. Two general-purpose VLMs were run through the identical pipeline on the identical crops: **Llama-4-Scout** and **Pixtral-12B**. A third (Gemini) was not run to completion because its free tier caps daily requests per model below the 200 crops required; it is not reported as a result. All naming crops (V4a and V4b, positive and control) are drawn from the held-out **test** split — `export_crops.py` and `export_tube_crops.py` both build their crop sets from `index.subset("test")`, never train or validation.

### 3.7 V5 — mitigation

Mitigation is **Deep Feature Reweighting (DFR)**: extract frozen penultimate features, draw a group-balanced sample over `(confound, label)` groups, fit a logistic-regression head (`sklearn.linear_model.LogisticRegression`, default `lbfgs` solver and L2 penalty, `max_iter=2000`) on standardized features, and swap only the final linear layer. Group balancing iterates over the four possible `(confound, label)` cells, keeps only the non-empty ones — chest tubes co-occur almost exclusively with pneumothorax-positives, so `(tube=1, label=0)` is structurally empty and excluded rather than balanced against — and subsamples the remaining non-empty groups to the size of the smallest one (`group_balanced_sample`, `dfr.py`). **ERM** and an **error-upweighted head retraining** baseline (labeled JTT after Liu et al. 2021, whose upweighting rule it follows, though the original method retrains the full network — here, as with DFR, only the frozen-feature logistic head is refit, so the numbers are not directly comparable to full-network JTT results in the literature) are run from the same feature-extraction path, over 3 seeds. Because none of these three methods import the naming module, this comparison doubles as the **naming-off ablation**: the loop's mitigation result cannot depend on the VLM. Metrics are worst-group accuracy (pneumothorax *without* the confound), in-distribution accuracy (pneumothorax *with* the confound), and the reliance gap.

A **linear probe** control freezes a shortcut-trained backbone and fits a fresh head on confound-free images, to test directly whether shortcut reliance hollows out or merely degrades core-feature learning.

### 3.8 V6 — cross-site generalization

The V1 model is evaluated unchanged on PadChest. Two mitigation variants are distinguished, and the distinction is the point of the spike:

- **Supervised target adaptation** — fit the last layer on *labeled* PadChest data. This tests adaptation, not shortcut mitigation: it would help even with no shortcut present.
- **Zero-shot source-DFR** — reduce the known source-side shortcut using NIH data only, touch **no** target labels, then evaluate PadChest. This is the test that can license a claim that the cross-site gap is shortcut-driven.

PadChest images are 16-bit while NIH images are 8-bit. Reconciling this at load time (rather than by changing the model's evaluation transform) is required for validity. Two checks validate the conversion: first, the **NIH reference AUC** is re-measured after the fix and confirmed to hold at its pre-fix value (0.842), which validates that the min-max rescaling function did not silently corrupt the pipeline it shares with the NIH path. Second, a direct pixel-intensity comparison — mean and per-image standard deviation over 100 randomly sampled NIH images versus 100 randomly sampled PadChest images, both passed through the same production `_open_as_8bit_L` function the model actually sees — shows the two distributions are closely matched: mean intensity 123.95 (NIH) vs. 124.27 (PadChest), a difference of +0.32 on a 0–255 scale, and per-image contrast (standard deviation) 60.51 (NIH) vs. 68.15 (PadChest) — PadChest is *not* lower-contrast after conversion; if anything it is measurably higher (ratio 1.126). This rules out the specific failure mode of concern (that global min-max rescaling could wash out tissue contrast or clip highlights in the higher-dynamic-range PadChest source) as a contributor to the measured OOD drop.

### 3.9 Reproducibility

Every run is config-driven (typed YAML with inheritance and CLI overrides), fixed-seeded, and logged to a timestamped directory containing the resolved config, `metrics.json`, and a checkpoint. Dataset paths are configuration values; no image or checkpoint is version controlled. The codebase is organized into confound-agnostic modules (`data/`, `train/`, `shortcut/`, `attribution/`, `discovery/`, `evaluation/`, `mitigation/`, `report/`) with 128 passing tests concentrated on the parts most likely to silently break: patient-disjointness of the loaders, the naming scorer's matchers (tested *not* to fire on generic anatomy), and the DFR group-balancing.

**Compute cost.** All local training/evaluation ran on a single consumer GPU (RTX 3050, 6GB); the core reported runs (V1/V2/V3 training and attribution, the 3-seed Phase-3 baseline table, V5, V6) total approximately 2 GPU-hours by `run.log` timestamp. The naming stage used 5,600 free-tier VLM API calls in total across the canonical V4a/V4b runs and every extension experiment (opacity sweep, occlusion study, multi-VLM and specialist arms); no generative model or fine-tuning step is used anywhere in the pipeline. This is the quantified basis for the compute-efficiency contrast drawn against RoentMod in Related Work.

**API-model reproducibility.** All Llama-4-Scout results were obtained via the Groq API in July 2026; the provider has since deprecated this model from its hosted roster. Exact reproduction of the Scout numbers by re-querying the API is therefore no longer possible — a limitation inherent to evaluating third-party-hosted VLMs rather than locally-weighted checkpoints. The raw prediction manifests (`predictions*.jsonl`) archived alongside this project are the artifact of record for these results, not a live, re-callable endpoint.

---

## 4. Experimental Design & Gates Summary

The project ran as six validation-gated spikes (V1–V6), each with a pre-registered pass/fail gate and a stop-and-report point. The table below summarizes the gates, measured outcomes, and status:

| Spike | Question | Gate | Measured | Status | Notes |
|-------|----------|------|----------|--------|-------|
| V1 | Trains end-to-end, sane AUC? | test AUC ≥ 0.75 | **0.8448** | **PASS** | in-distribution baseline |
| V2 | Model measurably cheats? | reliance gap >20pp | **44pp** | **PASS** | synthetic tag, p=0.95 |
| V3 | Saliency localizes confound? | peak-in-box >80% | **100%** | **PASS** | controls at 0.5% |
| V4a | VLM names SYNTHETIC confound? | P≥0.90 & R≥0.90 & halluc<10% | **P 0.978 / R 0.90 / halluc 0.02** | **PASS** | method works on easy target |
| V4b | VLM names REAL device? | P≥0.80 & R≥0.70 & halluc<10% | **P 0.57 / R 0.66 / halluc 0.50** | **FAIL** | **PIVOT TRIGGERED** |
| V5 | Mitigation recovers accuracy? | worst-group up, ID stable | **tube: 0.67→0.79, ID 0.96→0.96** | **PASS** | real confound, 3-seed mean |
| V6 | Real OOD gap exists + is shortcut-driven? | drop exists & narrows | **drop 0.192; zero-shot +0.005** | **PASS (gap) / REFRAMED (attribution)** | distribution shift, not shortcut-driven |

**V4b failure → pivot decision:** V4a passed at near-ceiling on an unambiguous synthetic confound, proving the *method* works. V4b failed on a real device confound specifically by *fabrication* (control-hallucination 0.50) — the failure mode the precision-over-recall gate was designed to catch. This triggered the pre-registered pivot: automated naming is withdrawn as a validated capability for real confounds, reported as a bounded negative result, and the project reframed as a detect→localize→mitigate study on both synthetic and real confounds without the VLM naming step as a claimed capability. The mitigation (V5) and cross-site (V6) results stand independently, since they never depend on the VLM.

**Post-pivot Phase 5 extension experiments** (§6 below) sharpened the V4b boundary across four dimensions: naming difficulty (opacity curve), image grounding (occlusion), specialist VLM performance, and five VLM-independent investigations into shortcut properties and robustness. All were run under the same validation discipline: report whatever the data shows, with no target outcome assumed.

---

## 5. Core Results (V1–V6)

The classifier trains to a sane baseline (V1: test AUC **0.8448**, validation 0.8269 — within the published NIH-pneumothorax range) and cheats measurably when given the opportunity (V2: reliance gap **44pp**; aligned 0.94, removed 0.50, flipped 0.05, measured on a class-balanced subset of 1,500 positives / 1,500 negatives). The removed-accuracy landing at exactly 0.500 — 750 correct out of the 1,500-per-class balanced set — is not a rounding artifact: it is the pure-shortcut fingerprint predicted and verified with a synthetic tag-detector *before* the run. A model conditioning almost entirely on the tag, once the tag is removed, collapses to a single constant prediction; on an exactly class-balanced evaluation set, a constant prediction is correct on exactly half the examples by construction. Attribution localizes the confound essentially perfectly (V3: peak-in-box **100%** on tagged positives vs **0.5%** on the same images untagged; mass fraction 0.14 in a box covering 0.8% of image area).

A companion **correlation-strength sweep** quantifies how this reliance gap emerges as the injected tag's correlation strength *p* increases from 0.6 to 0.95 (one training seed per point):

| *p* | trained reliance gap | worst-group (DFR before→after) | ID (DFR before→after) |
|---|---|---|---|
| 0.60 | 0.000 | 0.750 → 0.653 | 0.750 → 0.660 |
| 0.70 | 0.010 | 0.745 → 0.670 | 0.755 → 0.668 |
| 0.80 | 0.075 | 0.722 → 0.672 | 0.796 → 0.639 |
| 0.90 | 0.294 | 0.585 → 0.605 | 0.879 → 0.618 |
| 0.95 | 0.440 | 0.500 → 0.593 | 0.940 → 0.617 |

The reliance gap rises monotonically with *p*, with an **onset threshold**: below roughly p=0.7 the correlation is too weak for the model to exploit, and it falls back on real disease signal; above that threshold reliance climbs steeply, reaching the 44pp headline figure at p=0.95. DFR's in-distribution cost tracks the same trend — negligible where there is little shortcut to unlearn (p=0.6–0.7), and large where the shortcut dominates the representation (p=0.95, matching the standalone p=0.95 tag-DFR result in §6.4.2 above).

### 5.1 Naming succeeds on the synthetic confound and fails on the real one

**V4a passes at near-ceiling.** Blind-prompted Llama-4-Scout named the injected tag with precision **0.978**, recall **0.90**, and control-hallucination **0.02**, over 100 tagged positives and 100 confound-free controls. Recall landed exactly at its 0.90 bar on ten genuine model misses — an easy target already producing misses, which foreshadowed the harder case. The keyword matcher was corrected mid-analysis after a manual audit of all 23 original misses found that roughly 13 were the VLM correctly describing a "rectangular / opaque white object" in phrasing the initial regex did not cover; the fix was verified not to have relaxed the control-hallucination check, which remained at 0.02.

**V4b fails, and fails specifically on fabrication.** On real chest-tube crops the same model, same prompt, same pipeline yielded precision **0.57**, recall **0.66**, control-hallucination **0.50**. Detection and localization still worked on the real confound (tube reliance gap **28.75pp**; Grad-CAM side-consistency **96.6%**) — only naming broke. An audit of the control crops found **47 of 50** false positives explicitly naming a device ("endotracheal tube," "central line," "catheter") on images containing none: genuine hallucination, not a scorer artifact.

This is the pre-registered pivot condition (V4a passes, V4b fails), and it was executed as specified rather than iterated around: automated naming is withdrawn as a validated capability for real device confounds, and V4b is reported as a bounded negative result.

> **Figure 1** (`docs/paper/figures/figure1_naming_boundary.png`). The naming boundary: control-hallucination (device named on a confound-free crop) for each VLM on the synthetic confound versus the real chest tube. The boundary is **per-model**, not a flat clean split: Llama-4-Scout rises from **0.02** (synthetic) to **0.55** (real); Pixtral-12B rises from **0.31** (synthetic) to **0.83** (real). Both fabricate a device on a majority of clean control crops in the real-device setting; Pixtral already fabricates on nearly a third of clean synthetic-setting crops. The <10% gate is cleared only by Scout on the synthetic confound.

### 5.2 The failure is a property of general VLMs, not of one model

The obvious objection to a single-model negative is that a different VLM would not fail. We re-ran the **identical** 200 V4b crops, the **identical** blind prompt, and the **identical** `names_tube` matcher through two independent VLM families:

| model | precision | recall | control-hallucination |
|---|---|---|---|
| Llama-4-Scout | 0.545 | 0.660 | **0.550** |
| Pixtral-12B | 0.494 | 0.810 | **0.830** |

Both fabricate a device on a *majority* of clean control crops. The Scout re-run (0.550) reproduces the original V4b value (0.500) within run-to-run API and decoding drift; Pixtral is markedly worse. Audit of the raw text: Scout invents specific devices on clean anatomy ("endotracheal tube," "central venous catheter"); Pixtral emits near-identical boilerplate ("linear radiopaque object... looks like a medical device or foreign body") on **83 of 100** control crops. The matcher is conservative on positives — it does not fire on hedged "linear artifact" phrasing — which depresses recall in the safe direction rather than inflating hallucination.

An auditor built on off-the-shelf general-purpose VLMs would therefore raise a fabricated shortcut alarm on **55–83%** of clean images (Scout 55%, Pixtral 83%). This is precisely the false-alarm cost the precision-over-recall gate was designed to catch, now established as a cross-model property. A post-hoc bootstrap (10,000 resamples, seed 1337) confirms the synthetic-vs-real gap is statistically significant for both models: the 95% CI on the V4a-minus-V4b control-hallucination difference excludes zero (Scout −0.53 [−0.63, −0.43]; Pixtral −0.52 [−0.64, −0.40]), and Fisher's exact rejects independence (Scout p = 1.22e−18; Pixtral p = 7.56e−14). The result is independently corroborated by HalluCXR (arXiv:2605.20469), which reports 61.9–82.3% overall hallucination across six VLMs on 856 chest radiographs, with severity *highest on normal images* — the same qualitative signature, in a different model population on a different task.

### 5.3 What survives the naming failure

Mitigation, which does not depend on naming, holds on the real confound. DFR moves worst-group (pneumothorax-without-tube) accuracy **0.67 → 0.79 (3-seed mean; a single-draw run gives 0.83)** while in-distribution (pneumothorax-with-tube) accuracy is preserved **0.96 → 0.96**, narrowing the reliance gap **0.29 → 0.17**, using no test-distribution labels. DFR and JTT consistently outperform ERM across all three seeds — DFR worst-group **0.790 ± 0.036** vs. ERM **0.672 ± 0.022**, a margin of roughly 12 percentage points that exceeds either method's own seed-to-seed spread. We report this as an effect-size comparison rather than a significance test: a paired t-test at n=3 has virtually no power and an untestable normality assumption, so no p-value is reported here; the qualitative pattern — both alternatives beat ERM in *every* seed by a margin large relative to the observed variance — is the evidence, not a p-value derived from three points:

| confound | method | worst-group | ID | reliance gap |
|---|---|---|---|---|
| tube | ERM | 0.672 ± 0.022 | 0.959 ± 0.002 | 0.287 ± 0.025 |
| tube | JTT | 0.791 ± 0.047 | 0.940 ± 0.019 | 0.149 ± 0.031 |
| tube | DFR | 0.790 ± 0.036 | 0.964 ± 0.007 | 0.174 ± 0.029 |
| tag | ERM | 0.500 ± 0.000 | 0.940 ± 0.000 | 0.440 ± 0.000 |
| tag | JTT | 0.608 ± 0.018 | 0.531 ± 0.016 | −0.077 ± 0.025 |
| tag | DFR | 0.609 ± 0.022 | 0.603 ± 0.014 | −0.006 ± 0.024 |

> **Figure 2** (`docs/paper/figures/auditor_before_after_tube.png`). Before/after Grad-CAM on the real tube model, produced by a single auditor command (detect→mitigate). After DFR last-layer retraining, attention broadens away from the confound-associated region while in-distribution accuracy is preserved (0.96→0.97 on this single demo run) and worst-group accuracy rises.

The synthetic tag tells the cautionary half of the story: the gap closes (0.44 → 0.02) but in-distribution accuracy falls 0.94 → 0.62. A linear probe run specifically to falsify the tempting "the backbone is hollow" reading measured probe AUC **0.690** — above chance (0.50), below the clean baseline (0.845). Shortcut reliance **degrades but does not eliminate** core-feature learning, and the degradation is strength-dependent.

Cross-site, a real gap exists and is not explained by the shortcut. This comparison uses an **8,000-image subset of the full 25,600-image NIH test split**, capped deliberately for the V6 cross-site runner's compute cost (`run_v6.py`'s `max_per_split={"test": 8000}`, chosen because AUC is stable at this size) — distinct from V1's own headline test AUC (0.8448), which is measured on the complete 25,600-image split. On this 8,000-image subset, NIH reference AUC **0.842** falls to PadChest **0.650**, a drop of **0.192** (95% CI **[0.159, 0.225]**). NIH (n=8,000) and PadChest (n=5,289) are disjoint samples with no shared images or patients, so the two AUCs are bootstrapped **independently** — 2,000 stratified resamples drawn separately within each dataset — and the drop's confidence interval is the distribution of the per-resample difference; this is the standard approach for a difference between two independent samples, matching the magnitude reported by Zech et al. — a robust effect, not a borderline one. Supervised target adaptation recovers **+0.082** — but that is last-layer fitting on *labeled* PadChest and would help absent any shortcut, so it licenses no mitigation claim. The honest test, zero-shot source-DFR with no target labels touched, returns **+0.005** (95% CI **[−0.018, +0.028]**, same bootstrap procedure). The interval excludes a large recovery and includes zero, so the point estimate is not evidence of a real effect — but the interval is also wide enough that it does not tightly rule out a small-to-moderate one (up to roughly +0.03 AUC): this is an **underpowered null**, not a *tight* one, and we report it as such. The 0.192 drop is distribution shift whose cause this design cannot identify; shortcuts are one unproven candidate among scanner, population, label-pipeline, residual-intensity, and subset-composition differences.

---

## 6. Phase 5 Extension Experiments

After the V4b failure and pivot, five experiments were designed and run to sharpen the V4b boundary — i.e., to map *why* naming succeeds on a synthetic confound but fails on a real device. All were run under the same validation discipline: measure first, interpret second, no target outcome assumed.

### 6.1 Naming-Difficulty Curve: Visibility vs. Hallucination

**Working hypothesis (pre-experiment):** Fabrication rises as the confound gets fainter. We tested it directly by sweeping the synthetic tag's opacity across {0.25, 0.5, 0.75, 1.0} and re-scoring both models with the unchanged pipeline (n = 200 crops per cell, 100 positive + 100 control):

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

**Finding:** Reducing visibility produces a **recall cliff** — Scout's recall collapses 0.90 → 0.52 → 0.03 between opacity 0.75, 0.5 and 0.25 — while **control-hallucination stays flat**: ~0.02 for Scout and ~0.28–0.32 for Pixtral across *every* opacity level. Faintness makes the model miss the confound; it does not make the model invent one.

**Interpretation:** This **rules out visibility as the bridge** from V4a's 0.02 hallucination to V4b's 0.50–0.83. Whatever separates the synthetic tag from the real tube, it is not how hard the confound is to see. The revised working hypothesis is that fabrication tracks the **anatomical plausibility of the confound category**: a synthetic tag is never mistakable for anatomy at any opacity — it is either seen or not, with no ambiguous middle — whereas a real device sits inside a space of plausible near-misses (other lines, catheters, vessels, fissures, rib edges) that a VLM's medical-image priors can generate confident boilerplate about.

### 6.2 Image-Independence: Is Fabrication Grounded?

**Working hypothesis (pre-experiment):** The VLM is not reading the image at all — a live concern, since Lotfinia et al. (arXiv:2606.17710) show VLMs can answer chest-radiograph questions from language priors without attending to the image.

**Test:** Three per-crop conditions scored against each model's own unmodified-crop baseline description: occlude the **relevant** region, occlude an equal-area **irrelevant** region, or **swap** in a different same-kind crop. Verdict changes are broken down by direction: **pos→neg** (named at baseline, not named after) is consistent with grounding; **neg→pos** (a *new* device named after the edit) is fabrication merely relocated. All 12 cells at n = 100.

| dataset | condition | model | pos→neg | neg→pos | unchanged | change rate |
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

**Methodology caveat:** The flat gray fill used to occlude the tag is *itself* routinely described by the VLM as "a gray rectangular object... artificial and not part of the patient's anatomy... a label or marker" — language that trips the exact `names_confound` rule the tag was designed to trigger. Consequently, **V4a's `relevant` numbers understate true grounding**, and the V4a-vs-V4b comparison is invalid for this question — the measurement is biased toward `unchanged` for V4a in a way it is not for V4b.

**The evidence that carries the conclusion is the V4b-internal contrast.** V4b's matcher (`names_tube`) requires device-specific vocabulary — which a flat gray rectangle does not resemble the way it resembles a generic "artificial marker" — so its susceptibility to the artifact is far lower. Within V4b, both conditions use the same matcher on the same dataset, differing only in *where* the occluder lands:

- Occluding the **tube region** drops device-naming: pos→neg **61/100** (Scout), **66/100** (Pixtral).
- Occluding an equal-area **irrelevant** region drops it far less: **30/100** (Scout), **13/100** (Pixtral).

**Device-naming tracks the tube pixels.** "V4b fabrication is pure image-independent prior-emission" is not supported. The reconciled mechanism: the VLM carries a strong **device-present prior** *and* grounds it on the salient region — correctly reading a real tube when one is present, and **over-reading ambiguous normal anatomy as a device** when none is. This is **prior-driven over-reading of an ambiguous salient region**: neither blind recitation nor clean perception.

### 6.3 Specialist VLM: Over-Naming vs. Under-Naming Axis

**Question:** Does a domain-specialized radiology VLM move the naming failure boundary, or just shift which priors it emits?

**Finding 1: General VLMs over-name.** The V4b fabrication that triggered the pivot was a general-VLM behavior: on tube-free controls, Llama-4-Scout fabricates a device on **0.55** of clean crops and Pixtral-12B on **0.83**. qwen2.5-vl-7b — a *larger* general VLM — over-names just as hard: its V4b control-hallucination is **0.74**. Over-naming is not an artifact of small models; it persists at 7B on a fresh model family.

**Finding 2: Specialist under-names, but structurally.** CheXagent-2-3B does not answer in free text; it emits grounding tokens. Scored with a grounding-aware matcher, it clears V4a's precision handsomely (precision **0.9663**, recall **0.86**, control-hallucination **0.03**) but scores exactly **0.0 / 0.0 / 0.0** on V4b. This is not a perception failure: CheXagent produced a grounded label on **100/100** V4b tube-positive crops. It is a *taxonomy* failure. Its entire observed grounding vocabulary is `{foreign object, diaphragm, other lesion, artificial object, aortic enlargement, mediastinal}` — **no device-specific class exists** anywhere in its output space: nothing that says tube, catheter, line, drain, or wire. The specialist can localize the tube and cannot *name* it, because "name it as a device" is not an option its taxonomy affords.

This raises a natural follow-up: even without the vocabulary to say "tube," did CheXagent's grounding boxes at least land in the right place? NEATX provides no pixel bounding box for tubes (only the coarse L/R side label used throughout this project), so a precise overlap check isn't possible, but a side-check is: of the 100 V4b tube-positive crops, 88 have a single scorable side (`R`: 59, `L`: 29); the other 12 are bilateral (`side='BOTH'` in NEATX) and are excluded, the same way `peak_on_side` excludes them elsewhere, since a bilateral annotation has no single correct half to check a box against. Across those 88 crops, boxes labeled `foreign object` or `artificial object` (22 total, its closest approach to flagging something anomalous) land on the annotated side **14/22 (63.6%)** of the time — nominally above the 50% chance level, but a binomial test gives *p*=0.14 (95% CI [0.44, 1.0]), not distinguishable from chance at this sample size. Across *all* grounding boxes regardless of label (170 total, mostly `diaphragm`), side accuracy is **51.8%** — indistinguishable from chance. This does not rescue the taxonomy-failure reading; it reinforces it: CheXagent is not quietly localizing the tube under a euphemistic label, it is not tracking the tube's location at all.

**Finding 3: Caution on the over-clean reading.** "Specialist under-names, general over-names" is the axis, but it is not the *only* axis, and the general models are not uniformly safe wherever the specialist is silent. Even on the easy **synthetic** V4a controls, fabrication is model-dependent: Scout hallucinates on **0.02** of clean crops but Pixtral on **0.31** — a per-model fabrication *floor* that has nothing to do with domain specialization. The auditor's reliability therefore turns on the specific model on both axes at once — how much it over-names on clean crops *and* whether its vocabulary can name the target at all.

### 6.4 VLM-Independent Investigations

Five investigations probe the robustness of the core results without invoking the naming step. None is a headline; each either strengthens a claim by an independent route or bounds an alternative explanation.

#### 6.4.1 Investigation 1: Severity-Stratified Tube Reliance

**Setup:** Stratify the 94 overlapping pneumothorax-positive test images at the median bounding-box area. Within each tier, measure accuracy with vs. without the tube.

**Findings:**
- Accuracy is perfect (`1.0000`) whenever a tube is present, at *any* lesion size.
- Accuracy collapses when absent, at *any* lesion size: low-severity gap 0.400, high-severity gap 0.4375.
- The gap is large in both tiers, ruling out a pure disease-burden proxy.
- **Study is underpowered** (n=16–20/group) to distinguish true invariance from a real severity effect; the invariance claim is **withdrawn**. What stands is that even low-severity cases show a large gap — the shortcut is **device-keyed, not disease-burden-keyed**.

#### 6.4.2 Investigation 2: Canonical DFR Refit

**Setup:** Tune regularization C via 5-fold cross-validation, average weights over 10 balanced draws.

| confound | variant | worst-group | ID | gap |
|---|---|---|---|---|
| tube | before DFR | 0.6700 | 0.9575 | 0.2875 |
| tube | standard DFR | 0.8425 | 0.9650 | 0.1225 |
| tube | canonical DFR | **0.9275** | **0.9975** | **0.0700** |

**Finding (tube):** The canonical variant strictly dominates the standard one on all three metrics. The DFR result is a floor, not a ceiling — but the reported 0.79 (3-seed mean) is the number we defend most strongly.

| confound | variant | worst-group | ID | gap |
|---|---|---|---|---|
| tag | before DFR | 0.5000 | 0.9397 | 0.4397 |
| tag | standard DFR | 0.5927 | 0.5937 | 0.0010 |
| tag | canonical DFR | **0.6813** | **0.6870** | **0.0057** |

**Finding (tag):** Standard DFR (C=1.0) causes a catastrophic ID accuracy collapse (0.94 → 0.59); the gap nearly vanishes (0.001) but only because the model is now near-random overall, not because confound reliance was untangled. Canonical DFR (C=0.001, 10-draw) preserves the near-zero gap (0.0057) while recovering some ID accuracy (0.687), still well below the pre-DFR baseline (0.94). **This is not a data-loading bug — it is an intentional construction that reveals a real limitation.** Because the tag is a *synthetic, injectable* marker rather than a naturally-varying device, measuring both confound states requires building an untagged copy and a synthetically re-tagged copy of the *same* validation images and concatenating them — the only way to construct all four `(confound, label)` groups for a marker that can be painted onto any image at will (`extract_full_val_tag`, `canonical_dfr.py`; the same logic underlies V2's aligned/removed/flipped reliance-gap conditions). The consequence is real: pairing each image with itself under both confound states gives the logistic head no clean confound-vs-label axis to separate, and it collapses toward a coin-flip regardless of C or draw count. A plausible complementary reading is feature dominance rather than pure axis-indistinguishability — at p=0.95 the backbone's representation is so shaped by the tag that once the tag is decorrelated from the label by construction, the remaining disease-relevant signal is too weak relative to it to support accurate classification — but this project did not separately inspect feature magnitudes or head weights to adjudicate between the two framings, so both are offered as consistent with the observed collapse rather than either being independently confirmed. This is a boundary condition of linear last-layer reweighting specific to *paired-copy* synthetic confounds, not evidence that the pipeline is unreliable elsewhere — but it does mean the tag-DFR numbers above are uninterpretable as a measurement of DFR's mitigation efficacy. The tube result, not the tag result, is the one this project defends as a mitigation claim; the tag result stands only as a pipeline-validation and cautionary data point.

#### 6.4.3 Investigation 3: Site-Projection Re-Decodability

**Setup:** Fit a hospital-detector probe on raw features. Then project out the probed site direction and re-score disease AUC.

**Results:**
- Hospital detector: AUC **0.9993** (site is almost perfectly encoded)
- Disease AUC before projection: **0.6436** (measured on the full 7,572-image PadChest working set — this experiment calls `build_padchest_index` with `dfr_frac=0.0`, holding out no adaptation split — hence slightly below §5.3's 0.650, which is measured on the 5,289-image held-out test split only; same full-set-vs-held-out-split distinction as the zero-shot source-DFR arm in §3.2, not a new discrepancy)
- Disease AUC after projecting out site: **0.6436** (Δ **0.000**)
- Site-probe AUC on projected features: **0.9990** (site signal survives almost completely)

**Finding:** Site identity is redundantly encoded across many directions. A rank-1 projection is insufficient to ablate site information; a full subspace intervention would be required.

**Companion check: label-free test-time adaptation recovers none of the gap.** Two standard TTA methods were run on the frozen V1 model against PadChest, touching no target labels: **BN-adaptation** (re-estimating BatchNorm running statistics on the target images) moves AUC by **−0.004**, and **TENT** (entropy-minimizing the BN affine parameters) actively harms it (**−0.122**, collapsing toward chance). TENT was designed around multi-class categorical entropy; on this architecture's single-logit binary head, the entropy being minimized is a degenerate special case that can drive the logit toward an extreme, miscalibrated value rather than genuine adaptation. This result is consistent with that known architecture mismatch specifically, not a general claim about TENT's suitability for chest-radiograph adaptation in properly multi-class or calibrated setups. Together with the site-projection null above, this rules out the two mechanisms one would reach for first — a disease head riding a decodable site signal, and a covariate shift correctable by cheap label-free adaptation — leaving most of the 0.192 gap as genuine, unattributed distribution shift.

#### 6.4.4 Investigation 4: Label-Free OOD-Gap Attribution (Tag Source-DFR)

**Setup:** Generalize the zero-shot source-DFR test from the tube confound to the synthetic tag. Mitigate tag reliance on NIH (zero PadChest labels), then evaluate on PadChest.

| metric | value |
|---|---|
| NIH test AUC (tag model) | 0.7407 |
| PadChest AUC before mitigation | 0.6345 |
| PadChest AUC after tag source-DFR | **0.5520** |
| Zero-shot change | **−0.0825** |

**Finding:** Tag source-DFR does not improve PadChest AUC; it makes it substantially worse. The tag shortcut is so dominant that the feature space has largely collapsed to encoding the tag rather than disease. This asymmetry (vs. tube's +0.005 null) is consistent with the tag being a far more complete representation override. **Synthetic confounds prove the method but systematically overstate the severity of the phenomenon relative to real ones.**

---

## 7. Discussion

The paper's spine is a boundary: blind, general-purpose VLM naming works where a shortcut is unmistakable and fails — by fabrication — where a shortcut is a plausible member of a familiar category. The two mechanism experiments locate *which* axis the boundary runs along. It is not visibility: faintness produces a recall cliff, not a hallucination cliff (§6.1). It is plausibility. A synthetic corner tag is never mistakable for anatomy at any opacity, so the model either sees it or does not; a chest tube lives among lines, catheters, vessels, fissures, and rib edges that a device-primed VLM can write confident boilerplate about. §6.2 then rules out the deflationary reading that the VLM ignores the image: device-naming tracks the tube pixels (occluding the tube drops naming roughly twice as much as occluding an equal-area irrelevant region). The failure is therefore neither blind recitation nor clean perception — it is **prior-driven over-reading of an ambiguous salient region**.

We frame the failure as an **over-name / under-name axis**, because the two directions carry different costs for an auditor. Under-naming (a miss) degrades gracefully — §6.1 shows that as the confound fades the model simply stops naming it, and the recall bar is set below precision precisely to tolerate this. Over-naming (fabrication) is the dangerous direction: a tool that invents a chest tube on a clean radiograph will send a clinician chasing a shortcut that is not there, and at 55–83% on clean crops it would do so on a majority of images. This is why the design made control-hallucination a *gate* rather than a reported statistic, and why the honest verdict is that off-the-shelf general VLMs are not safe as blind auditors of clean images without such a gate. That the boundary is **per-model** rather than a flat clean split (Scout fabricates on 2% of synthetic controls but Pixtral on 31%) sharpens rather than softens the conclusion: even in the "easy" synthetic setting, one of two general models already fails the gate.

What survives the naming failure matters for how the contribution should be read. The mitigation result is real, defensible, and — by construction — independent of the broken naming step: no method in the mitigation path imports the VLM module, so the naming-off ablation is not a separate experiment but a property of the code. DFR recovers worst-group accuracy on the real tube from 0.67 to 0.79 (3-seed mean) with in-distribution accuracy preserved and no target-distribution labels, and two independent methods agree well outside seed variance. We are deliberate about the headline number: 0.79 is the 3-seed mean; the single-draw 0.83 appears only when explicitly labeled as such, because reporting the luckier draw as the result would be the same over-claiming the paper's method is designed to catch. The synthetic tag supplies the cautionary companion — a pure shortcut at p=0.95 so thoroughly overrides the representation that removing it leaves too little disease signal behind, which is exactly why synthetic confounds prove the *method* but overstate the *phenomenon*.

The cross-site result is where we most resist the tempting story. There is a real 0.192 US-to-Spain AUC drop, Zech-magnitude, and it would be easy to present the loop as "validated cross-site." It is not. The supervised recovery (+0.082) uses PadChest labels and would help with no shortcut present; the honest zero-shot source-DFR test — reduce the *known* tube shortcut on NIH only, touch no target labels — returns +0.005 (95% CI [−0.018, +0.028]): a point estimate indistinguishable from zero. Reducing a known source shortcut did not measurably narrow the cross-site gap in this test, so the gap is distribution shift this design cannot attribute to the shortcut — while acknowledging the test's power to detect a modest effect, had one existed, was limited. The representation-level controls corroborate rather than rescue this: site identity is almost perfectly decodable (AUC 0.9993) yet the frozen disease head does not read it (Δ 0.000 after projecting site out), so the two mechanisms one reaches for first — a head riding a site signal, and a covariate shift correctable by label-free adaptation — are both bounded near zero, leaving most of the gap genuinely unattributed. The clinically load-bearing consequence is that "fix the known shortcuts" is *not* sufficient due diligence before cross-site deployment.

Finally, honest scope. This is one pathology (pneumothorax), one primary corpus (NIH) with one second site (PadChest), one real device confound (chest tube) with non-expert labels, and general (not specialist) VLMs. The two mechanism experiments bound the boundary on the visibility and image-grounding axes but do not map it along the plausibility axis the mechanism implicates — we know naming works on an unmistakable marker and fails on a subtle device, not where in between it turns over. The pre-registered stopping rule governed the gate decision itself — a V4b failure after a V4a pass triggered an immediate, reported pivot rather than iteration on that decision — but it did not foreclose a cheap follow-up ablation (abstention prompting), which was designed and coded but not executed before this writeup. The boundary is therefore *characterized*, not *mapped*, partly by design and partly because that follow-up remains unrun.

---

## 8. Limitations

**Single pathology, single primary corpus.** Every result concerns pneumothorax on NIH ChestX-ray14 and its cross-site evaluation on PadChest. The mechanism proposed in §6 — fabrication driven by the anatomical plausibility of the confound category — predicts that other device-adjacent pathologies would behave similarly, but this is a prediction, not a measurement.

**Non-expert ground truth for the real confound.** NEATX's chest-drain annotations come from two data-science students, not radiologists, with "moderate to almost perfect" agreement against expert labels reported in the source paper. This is adequate for inducing a training-time enrichment signal and defining a mitigation target, but it is a noisier ground truth than a radiologist-drawn segmentation would provide for V4b's precision/recall scoring specifically.

**The auditor's own VLM component was not tested for pure image-independence in V4a.** Lotfinia et al. (arXiv:2606.17710) show VLMs can answer CXR questions from language priors without attending to the image. §6.2 addresses this and does *not* support blind recitation as the explanation of V4b (device-naming tracks the tube pixels), but a clean counterfactual-image test on V4a specifically — free of the flat-gray-occluder artifact documented in §6.2 — remains the most direct follow-up. That same occluder artifact also means the V4a-vs-V4b grounding comparison in §6.2 is not directly available: V4a's `relevant`-occlusion numbers are contaminated by the occluder itself tripping the `names_confound` matcher, so the image-independence conclusion rests entirely on the V4b-internal contrast (relevant vs. irrelevant occlusion, same matcher, same dataset). This is sufficient to support the grounding claim made, but it means no experiment in this project directly compares V4a's and V4b's degree of grounding to each other.

**The V4b matcher's false-negative rate has been audited; a small, quantified vocabulary gap exists but does not change the verdict.** A manual audit of all 34 V4b recall misses (mirroring the V4a 23-miss audit) found: **2 of 34 (5.9%)** are confirmed matcher/regex bugs — the VLM explicitly and unambiguously identified a device ("...appear to be **wires**, likely from an ECG or other monitoring device"; "...likely to be **medical devices**, specifically the patient's ECG **leads**"), but the `names_tube` regex matches singular "wire"/"lead"/"medical device" and not their plural forms. **1 of 34** shows the VLM detecting a real but different device (a "pacemaker or implantable cardioverter-defibrillator," not the annotated chest tube). The remaining **31 of 34 (91%)** are genuine VLM perception failures — explicit "no artificial/foreign objects, normal anatomy" statements, or tube-like linear structures misattributed to ribs, the imaging grid, or clothing. Correcting the two confirmed regex bugs would raise V4b recall from 0.66 to at most 0.68 — still below the 0.70 gate, so the FAIL verdict is unchanged. The control-hallucination result (0.50, the metric that actually drove the pivot) is unaffected either way, since control-hallucination requires the matcher to *fire* on a negative crop, not to miss on a positive one, and this audit only examined misses.

**Two-site validation; V6 causal attribution narrowed but not closed.** Two sites are two points, not a trend. The cross-site drop (0.192) is real and Zech-magnitude, and the §6.4.3 controls rule out two candidate mechanisms — a disease head riding a decodable site signal (Δ 0.000) and label-free covariate shift correctable by standard test-time adaptation (BN-adaptation −0.004; TENT −0.122, actively harmful). But most of the gap remains genuine, unattributed distribution shift. Candidate causes this design cannot separate include scanner and acquisition differences, population differences, the two datasets' different label-generation pipelines (NIH's NLP-only labeler vs. PadChest's NLP-plus-manual process), residual intensity artifacts despite the 8-bit fix, and the composition of the 573-positive / 7,000-negative working subset. A third site would turn two points into a trend; the design accommodates one without rework, but we did not run it.

**Single-GPU scale; single-seed sweep.** All training ran on one consumer GPU (RTX 3050, 6GB), which shaped batch size (16, after an OOM at 32), resolution (224px, standard for this literature), and the decision to work with subsets (most visibly the 7,572-image PadChest working set). The correlation sweep carries one *training* seed per point; its trend (monotone reliance growth) is clear, but individual point values could shift under a different seed. The 3-seed baseline table is the result we defend most strongly on variance grounds.

**The difficulty boundary is bounded on one axis, unmapped on the others.** §6.1 rules out visibility as the axis along which naming degrades into fabrication, and the multi-VLM replication (§5.2) closes the "a different model would fix it" escape hatch for general-purpose VLMs (55% and 83% fabrication on clean crops). What remains open is where between an unambiguous synthetic marker and a subtle real device the boundary sits along the *plausibility* axis, and whether prompt design, crop resolution, or abstention/confidence mechanisms shift it. An abstention-prompt ablation targeting exactly this was designed and implemented but not executed before this writeup; we report it as built-but-unrun rather than as a considered decision not to attempt a rescue.

**Interpretive claims corrected mid-project.** Two claims were written before the control that tests them existed, and both were withdrawn once it was run: "the pure-shortcut backbone is hollow" (falsified by the linear probe: AUC 0.690, above chance 0.50) and "mitigation validated across three settings" (falsified by the zero-shot source-DFR null: +0.005, after the +0.082 "recovery" was correctly re-identified as supervised target adaptation). We report both, because the failure mode they share — asserting an interpretation the experiment does not license — is the same one this project's method is designed to catch in a classifier.

---

## 9. Conclusion

We set out to build an auditor that detects, names, and mitigates a spurious shortcut in a chest X-ray classifier, and we ran it under a validation-gated discipline that let each stage's claim stand or fall on a pre-registered threshold. Detection, localization, and mitigation transfer from a synthetic confound to a real chest-tube confound: reliance is measurable, attributable, and reducible — worst-group accuracy on untreated (tube-free) pneumothorax rises from 0.67 to 0.79 with no target labels, on a step that provably never touches the VLM. The naming stage does not transfer. Blind, general-purpose VLM naming passes at near-ceiling on an unmistakable synthetic marker and fails on the real device by *fabrication*, inventing chest tubes and catheters on a majority of clean radiographs — a cross-model failure, corroborated by independent benchmarks, that our control-hallucination gate was built to catch and did. And the real cross-site drop we measured is distribution shift we could not attribute to the shortcut, not a mitigation success.

The contribution we stand behind is therefore not "an auditor that names shortcuts," but a mapped boundary of where that ambition currently breaks and a defensible mitigation result that survives the break. Blind VLM naming is a bounded capability — trustworthy for salient, unambiguous shortcuts, unsafe as a blind auditor of clean images without a false-alarm gate. The most useful thing a capstone can do with a failed critical gate is report it precisely, characterize why it fails, and show what still works around it. That is what this project does: an honest negative result on generative-model-free shortcut naming, wrapped in a mitigation and cross-site study that neither overstates the recovery nor hides the failure. The natural next steps — a radiologist-grade real-confound corpus for re-scoring the gate, a specialist-VLM arm to test whether radiology fine-tuning moves the boundary, and a third site to turn two cross-site points into a trend — are all accommodated by the confound-agnostic design without rework.

---

## Ethics and Data/Code Availability

This study used publicly available, de-identified datasets (NIH ChestX-ray14, PadChest, NEATX). No patient-identifiable information was accessed; institutional review was not required for secondary analysis of public data.

Code, configuration files, and prediction manifests for the reported experiments are available upon reasonable request to the authors.

---

## References

- Zech, Badgeley, Liu, Costa, Titano, Oermann (2018). *Variable generalization performance of a deep learning model to detect pneumonia across hospital systems.* PLOS Medicine. arXiv:1807.00431.
- DeGrave, Janizek, Lee (2021). *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence.
- Jiménez-Sánchez, Juodelyte, Chamberlain, Cheplygina (2023). *Detecting Shortcuts in Medical Images: A Case Study in Chest X-rays.*
- RoentMod (2025/2026). *A synthetic CXR modification model to identify and correct image-interpretation shortcuts.* npj Digital Medicine. arXiv:2509.08640.
- Sagawa, Koh, Hashimoto, Liang (2020). *Distributionally Robust Neural Networks (GroupDRO).* arXiv:1911.08731.
- Liu, Haghgoo, Chen, Raghunathan, Koh, Sagawa, Liang, Finn (2021). *Just Train Twice (JTT).* arXiv:2107.09044.
- Kirichenko, Izmailov, Wilson (2022). *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations (DFR).* arXiv:2204.02937.
- Zhao, Hauke, De Castilhos, Bode, Kather, Nebelung, Truhn (2026). *From Clinical Intent to Clinical Model.* arXiv:2604.17110.
- Ghosh, Syed, Wang, Choudhary, Li, Poynton, Visweswaran, Batmanghelich (2025). *LADDER: Language-Driven Slice Discovery and Error Rectification.* Findings of ACL 2025. arXiv:2408.07832.
- Hosseini, Nawathe, Moayeri, Balasubramanian, Feizi (2025). *SpurLens.* arXiv:2503.08884.
- Varma, Delbrouck, Chen, Chaudhari, Langlotz (2024). *RaVL.* NeurIPS 2024. arXiv:2411.04097.
- Fay, Reguigui, Yang, Gatidis, Küstner (2025). *MIMM-X.* arXiv:2511.22990.
- Cheplygina, Damgaard, Eriksen, Juodelyte, Jiménez-Sánchez (2025). *NEATX: Augmenting Chest X-ray Datasets with Non-Expert Annotations.* MICCAI MIUA 2025. arXiv:2309.02244.
- Wang & Li (2026). *HalluCXR.* arXiv:2605.20469.
- Lotfinia, Ziegelmayer, Adams, Truhn, Maier, Arasteh (2026). *Vision-language models for chest radiography do not always need the image.* arXiv:2606.17710.
- CXR-LLaVA (2023). arXiv:2310.18341. LLaVA-Rad (Microsoft).
- Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra (2017). *Grad-CAM.*

---

## Appendix A — Reproducibility & Compute Cost

### A.1 Configuration & Environment

Every run is config-driven (typed YAML with inheritance and CLI overrides), fixed-seeded, and logged to a timestamped directory. Dataset paths are configuration values; no image or checkpoint is version controlled. The codebase is organized into confound-agnostic modules with 128 passing tests.

**Python version:** 3.9.13  
**PyTorch + CUDA:** torch 2.4.0 on CUDA 12.1, timm (vision models)  
**Hardware:** RTX 3050 6GB (Windows 11)  
**Reproducibility:** `git rev-parse HEAD` recorded in every run; all random seeds fixed (numpy, torch, Python)

### A.2 Compute Cost

Extracted from `run.log` timestamps in experiment directories:

| run | wall-clock | notes |
|---|---|---|
| V1 training (DenseNet-121, 5 ep, full split) | 45.1 min | |
| V2 shortcut training | 34.2 min | |
| V3 Grad-CAM attribution (200 images) | 21.1 min | |
| Phase 3 baselines (ERM/JTT/DFR × 3 seeds × 2 confounds) | 16.6 min | |
| V5 DFR fit (single run) | 1.4 min | |
| V6 cross-site eval | 2.6 min | |
| **Total (core reported runs)** | **~2.0 GPU-hours** | single RTX 3050 |

Not included: correlation-strength sweep (5 training runs, ~2.5–4 hours), V4 crop export (much smaller), exploratory/failed runs.

**Total estimated local compute:** 2–6 GPU-hours on a single consumer laptop GPU.

### A.3 VLM API Calls

Counted from `predictions*.jsonl` line counts:

| study | calls |
|---|---|
| V4a + V4b canonical | 400 |
| Multi-VLM replication | 400 |
| Opacity sweep (Exp 1) | 1,600 |
| Occlusion study (Exp 2) | 1,200 |
| Specialist arm (Exp 3) | 2,000 |
| **Total** | **5,600 free-tier calls** |

All against free-tier APIs (Groq, Mistral La Plateforme, Google AI Studio); zero marginal cost, no generative model or fine-tuning at any point.

---

## Appendix B — Statistical Hardening

Post-hoc bootstrap statistics over existing prediction files (no new runs).

### B.1 Control-Hallucination Rate per Cell

| dataset | model | flagged / n | point estimate | 95% CI (bootstrap percentile) |
|---|---|---|---|---|
| V4a | scout | 2 / 100 | 0.02 | [0.00, 0.05] |
| V4b | scout | 55 / 100 | 0.55 | [0.45, 0.65] |
| V4a | pixtral | 31 / 100 | 0.31 | [0.22, 0.40] |
| V4b | pixtral | 83 / 100 | 0.83 | [0.75, 0.90] |

### B.2 V4a-vs-V4b Gap, per Model

| model | gap (V4a − V4b) | 95% CI on difference | Fisher's exact p (two-sided) |
|---|---|---|---|
| scout | −0.53 | [−0.63, −0.43] | 1.22e−18 |
| pixtral | −0.52 | [−0.64, −0.40] | 7.56e−14 |

For both models, the 95% bootstrap CI on the V4a-minus-V4b control-hallucination difference excludes 0, and Fisher's exact test rejects independence at the 0.05 level. The gap is statistically significant.

**Methodology:** Non-parametric bootstrap, percentile method, 10,000 resamples, seed 1337. Gap bootstrapped directly (V4a and V4b resampled independently per resample).

---

## Appendix C — Decision Log (abbreviated from PROGRESS.md)

**2026-07-07 — Framework & compute:** PyTorch + timm (DenseNet-121 @ 224px), local RTX 3050 6GB, AMP on, batch-size tuned.

**2026-07-08 — V1 PASS:** Full-data test AUC 0.8448. Root-caused CUDA memory failure to VRAM peak on AdamW; fixed by batch_size 32→16.

**2026-07-08 — V2 PASS:** Reliance gap 44pp. acc_removed landing at 0.500 matches pure-shortcut fingerprint pre-validated before run.

**2026-07-08 — V3 PASS:** Grad-CAM peak-in-tag-box 100% on tagged positives vs 0.5% on untagged controls.

**2026-07-08 — V4a PASS (CRITICAL GATE):** Llama-4-Scout via Groq named synthetic tag, P 0.978 / R 0.90 / halluc 0.02. Keyword matcher corrected during scoring after auditing all 23 original misses.

**2026-07-08 — V4b FAIL → PIVOT:** Real chest-tube confound. Detect/localize succeeded (gap 28.75pp). Naming failed: P 0.57 / R 0.66 / halluc **0.50**. Llama-4-Scout fabricates devices on 50/100 tube-FREE controls (47 of those 50 audited as explicit, genuine hallucination, not matcher error). Pivot executed: auto-naming dropped; reframe as detect→localize→mitigate without naming claim.

**2026-07-09 — V5 PASS (DFR):** REAL tube confound: worst-group 0.67→0.79 (3-seed mean; a single-draw run gives 0.83), ID 0.96→0.96 (3-seed mean). Synthetic tag (p=0.95): gap closes but ID collapses 0.94→0.62. Linear probe (AUC 0.690) shows backbone is degraded, not hollow.

**2026-07-09 — Phase 3:** ERM/JTT/DFR baselines (3-seed table), naming-off ablation, correlation-strength sweep (monotone gap vs p), GroupDRO skipped.

**2026-07-10/11 — V6:** NIH 0.842 → PadChest 0.650 (drop 0.192). Supervised adapt +0.082. Zero-shot source-DFR +0.005 (null). Drop is distribution shift, not shortcut-driven.

**2026-07-11 — CO1-CO4 writeup complete.** Literature re-search, novelty framing. Two corrected claims (hollow-shortcut, cross-site mitigation).

**2026-07-14–15 — Phase 5 Exp 1–3:** Opacity sweep, occlusion study, specialist VLM (over-name / under-name axis).

**2026-07-16–18 — Phase 5 Investigations 1–4:** Severity-stratified gap, canonical DFR, site-projection, label-free OOD-gap attribution.

**All experiments complete. Capstone writeup assembled.**

---

*End of capstone writeup. For detailed experimental record, see PROGRESS.md. For project operating principles, see CLAUDE.md.*
