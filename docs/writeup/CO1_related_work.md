# CO1 — Problem, Related Work, and Novelty Framing

*Spurious-Correlation Auditor for Chest X-ray Classifiers.* This document maps the problem to
prior art and states where our contribution sits. It is written against the **final measured
results of the complete validation gauntlet (V1–V6)**, including one failed gate (V4b, triggering
a documented pivot) and two claims that were corrected after a targeted review caught them
asserting more than the data supported (see §4, §6). Literature was re-checked exhaustively
through July 2026 before finalizing this document.

## 1. The problem

Deep chest X-ray (CXR) classifiers routinely reach high in-distribution accuracy while relying
on **spurious shortcuts** — features correlated with the label in the training data but not
causally related to the pathology (scanner text, laterality/positioning markers, hospital- or
device-specific artifacts). Such models appear accurate yet fail under distribution shift
(a new hospital, scanner, or population), which is a patient-safety problem: a model that keys
on a chest-tube's presence rather than the pneumothorax it treats will silently mispredict
where that correlation breaks.

The auditing task has four parts, and prior work tends to address only some:
1. **Detect** that a classifier relies on a shortcut.
2. **Name** the shortcut in human/clinical terms — without being told what to look for.
3. **Mitigate** the reliance.
4. **Prove recovery** under out-of-distribution / cross-site shift.

Our contribution centers on step (2) — *prompt-free, generative-model-free naming* — wired into
a full detect→localize→name→mitigate loop, evaluated on a synthetic confound (proof of method),
a real confound (chest tube, proof of transfer attempt), and real cross-site data (proof of
generalization-gap measurement).

## 2. Related work, grouped

### 2a. Evidence that CXR shortcuts are real and dangerous
- **Zech et al., 2018 (PLOS Medicine)** — *Variable generalization performance of a deep learning
  model to detect pneumonia.* CNNs trained across NIH / Mount Sinai / Indiana generalized worse
  externally than internally, and could identify the **hospital system with ~99.95–99.98%
  accuracy** — direct evidence that site-specific confounders drive predictions.
  ([PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683),
  [arXiv:1807.00431](https://arxiv.org/abs/1807.00431))
- **DeGrave, Janizek & Lee, 2021 (Nature Machine Intelligence)** — *AI for radiographic COVID-19
  detection selects shortcuts over signal.* Using explainable-AI methods, showed COVID-19 CXR
  models relied on **text markers and patient positioning** rather than pathology, and argued
  explainable AI is a *prerequisite* to clinical deployment.
  ([Nature MI](https://www.nature.com/articles/s42256-021-00338-7))
- **Jiménez-Sánchez, Juodelyte, Chamberlain & Cheplygina, 2023** — *Detecting Shortcuts in Medical
  Images: A Case Study in Chest X-rays.* Manual/statistical shortcut-detection case study on
  chest X-rays, predating VLM-based naming — same genre as Zech/DeGrave (detects, does not name
  via a general-purpose model).

*These motivate the project and detect shortcuts by hand or by statistical association — none
automatically **name** a shortcut in clinical language via a general-purpose model.*

### 2b. Correcting shortcuts by generative counterfactuals (the closest prior art)
- **RoentMod, 2025 (npj Digital Medicine; arXiv:2509.08640)** — *A synthetic CXR modification
  model to identify and correct image-interpretation shortcuts.* Combines a medical image
  generator (**RoentGen**) with an image-to-image editor to produce **counterfactual CXRs** with
  user-specified findings while preserving anatomy; training on these counterfactuals improved
  discrimination by **3–19% AUC internally, 1–11% externally**. Reader studies rated 93% of
  images realistic. ([npj Digital Medicine](https://www.nature.com/articles/s41746-026-02497-6),
  [arXiv](https://arxiv.org/abs/2509.08640)) No citing follow-up work was found through July 2026.

*This is the method we most directly contrast against (see §6): it is powerful but relies on a
**generative diffusion pipeline** (RoentGen + editing) — heavy compute, model fine-tuning, and a
generative apparatus we deliberately avoid.*

### 2c. Mitigation baselines (worst-group robustness)
- **GroupDRO (Sagawa et al., 2020)** — distributionally-robust optimization over predefined
  groups; optimizes worst-group loss. Requires group labels at train time. (arXiv:1911.08731)
- **JTT — Just Train Twice (Liu et al., 2021)** — upweight examples an ERM model gets wrong;
  approximates group robustness without explicit group labels. (arXiv:2107.09044)
- **DFR — Deep Feature Reweighting (Kirichenko et al., 2022)** — retrain **only the last layer**
  on a group-balanced set. Reaches **92.9% worst-group on Waterbirds, 70.1% on CivilComments**,
  matching/beating GroupDRO at a fraction of the cost; premise is that the backbone already
  encodes core features and only their weighting is wrong.
  ([arXiv:2204.02937](https://www.alphaxiv.org/overview/2204.02937v2)) Later work questions
  whether last-layer retraining is always sufficient
  ([arXiv:2308.00473](https://arxiv.org/abs/2308.00473)), and a follow-up extends DFR to multiple
  layers ("Not Only the Last-Layer Features," [arXiv:2409.14637](https://arxiv.org/pdf/2409.14637)).
  Our own V5/sweep results are directly consistent with the arXiv:2308.00473 caution: on a
  *pure*-shortcut backbone (p=0.95), DFR's in-distribution recovery is incomplete — see §6.
- **"From Clinical Intent to Clinical Model" (Zhao, Hauke, De Castilhos, Bode, Kather, Nebelung,
  Truhn — arXiv:2604.17110, May 2026)** — an autonomous LLM coding-agent pipeline that, among
  other tasks, reduces chest-drain reliance for pneumothorax classification on NIH-derived data,
  reporting **60%→31% and 50%→18%** shortcut-reliance reduction on two datasets. This is the
  closest *concurrent* work to our V5 real-tube mitigation: same target shortcut (chest drain /
  pneumothorax), different mechanism (agentic pipeline search vs. our DFR last-layer retrain),
  different metric (% reliance vs. our worst-group accuracy / reliance gap). It does not report
  a VLM-naming step, control-hallucination, or cross-site transfer test. We cite it as an
  independent, contemporaneous validation that this specific real-world shortcut is a live
  target for the field, reached by a different route.

*These are our V5 mitigation candidates. DFR is the natural first choice on this hardware:
cheap, last-layer-only, strong worst-group gains, and it doubles as a named baseline alongside
ERM and JTT in our Phase 3 comparison table.*

### 2d. Foundation-model shortcut/spurious-correlation auditing (closest analogues to our naming step)
- **LADDER (Ghosh, Syed, Wang, Choudhary, Li, Poynton, Visweswaran, Batmanghelich — Findings of
  ACL 2025, arXiv:2408.07832)** — *Language-Driven Slice Discovery and Error Rectification in
  Vision Classifiers.* The **closest prior art to our naming step**, and the one we distinguish
  most carefully: LADDER uses an LLM's latent domain knowledge to hypothesize a classifier's
  biases **without pre-specified attribute labels**, and evaluates on radiology (**NIH-CXR**)
  alongside mammography (RSNA, VinDr) and natural-image benchmarks (Waterbirds, CelebA,
  MetaShift). Three differences make it a genuine neighbor rather than the same method: (i) **how
  the image is read** — LADDER *captions* images to text (BLIP / GPT-4o) and projects model
  activations into text via retrieval, then reasons with a **text-only LLM** over those sentences;
  we instead feed the **Grad-CAM crop pixels directly to a general vision-language model**, with
  no captioner, no report corpus, and no activation-to-text projection. (ii) **What is evaluated**
  — LADDER scores discovery via slice Precision@10 and reports downstream **worst-group accuracy**
  after error rectification; it has **no false-alarm / control-hallucination gate**. Our
  contribution is precisely to score the *named* confound on precision/recall **and** hallucination
  on confound-free control crops, because for an auditor a fabricated shortcut is worse than a
  missed one. (iii) **What question is asked** — LADDER's discovery is in service of mitigation;
  our naming step is evaluated as an end in itself ("can a blind VLM be trusted to name this
  shortcut?") and reports a **hard failure** on the real device confound (V4b), a negative result
  LADDER's mitigation-oriented evaluation is not designed to surface.
- **SpurLens (Hosseini, Nawathe, Moayeri, Balasubramanian, Feizi — arXiv:2503.08884, 2025)** —
  uses GPT-4 plus open-set object detectors to automatically discover spurious visual cues that
  multimodal LLMs over-rely on in natural images, and separately measures **hallucination
  amplification** (objects hallucinated up to 10× more often when a spurious cue is present).
  This is the closest *general-domain* analogue to our naming step: prompt-free-ish, automated
  discovery of shortcuts a model relies on. It does not operate on medical images, does not use
  a localize-then-blind-caption pipeline scored on precision/recall/control-hallucination against
  a clinical vocabulary, and does not touch chest X-rays or devices.
- **RaVL (Varma, Delbrouck, Chen, Chaudhari, Langlotz — Stanford AIMI, NeurIPS 2024,
  arXiv:2411.04097)** — discovers *region-level* spurious correlations inside a fine-tuned VLM's
  own classification errors via clustering, and mitigates with a region-aware loss (+8.2%
  worst-group accuracy, 191% discovery improvement over prior methods, across 654 VLMs). This is
  the closest general-domain analogue to the *combined* discover+mitigate loop, from a group with
  radiology adjacency (Stanford AIMI). It audits a VLM's own internal errors rather than using a
  second, independent VLM to caption a separate classifier's saliency region; it is not evaluated
  on chest X-rays.
- **MIMM-X (Fay, Reguigui, Yang, Gatidis, Küstner — arXiv:2511.22990, 2025)** — disentangles a
  **pre-specified** confounding factor (sex, age, acquisition site) from a diagnostic signal via
  mutual-information objectives, applied to CheXpert pleural effusion and brain-MRI age
  prediction. Requires the confounder to be **named in advance** — the opposite of our blind
  design — and does not use a VLM. Cited as background showing the standard alternative
  (disentanglement) still assumes foreknowledge of the confound.
- Grad-CAM-style attribution (Selvaraju et al.) localizes *where* a model looks — our V3 uses it
  — but does not **name** what it found.

*Exhaustive search (through July 2026) found LADDER (above) to be the nearest medical-imaging
analogue — LLM-driven, label-free bias naming evaluated partly on chest X-rays — but no paper
that (a) feeds a chest-X-ray classifier's **attribution crops** to a **general-purpose,
blind-prompted VLM** (rather than captioning them to a text LLM) and (b) scores the resulting
name against ground truth on a **precision/recall + control-hallucination** metric. This specific
niche — pixel-level blind naming, gated by a false-alarm-averse metric — remains open; LADDER is
the closest, and differs on both (a) and (b).*

### 2e. VLM reliability on chest radiographs (bears directly on our V4a/V4b results)
- **HalluCXR (Wang & Li, King's College London — arXiv:2605.20469, May 2026)** — benchmarks 6
  VLMs on 856 chest radiographs, finding **61.9–82.3% overall hallucination rates** and
  **36.4–80.2% "clinically dangerous" hallucination rates**, with the striking finding that
  hallucination severity is **highest on normal/control images** (51.9% severity-3 on normals).
  This is strong independent corroboration of our V4b control-hallucination rate (0.50 on clean
  tube-free crops) — the same qualitative failure mode, measured on a different VLM population
  and a different task (diagnostic findings vs. our device-naming).
- **"Vision-language models for chest radiography do not always need the image" (Lotfinia,
  Ziegelmayer, Adams, Truhn, Maier, Arasteh — arXiv:2606.17710, June 2026)** — shows VLMs can
  answer CXR questions correctly from language priors alone, without attending to the image. This
  is a direct limitation for our own method: the auditor's VLM-naming step could in principle be
  pattern-matching on typical radiology-report phrasing rather than the crop's actual content. We
  did not test for this and flag it explicitly in CO4 §4.
- GPT-4V-specific evaluations on chest radiographs (European Radiology PMC11914349; Radiology
  DOI 10.1148/radiol.233270) independently report >40% diagnostic-hallucination rates,
  reinforcing that high hallucination on CXRs is a documented, general property of off-the-shelf
  VLMs, not an artifact of our specific model (Llama-4-Scout) or prompt.
- **CXR-LLaVA (arXiv:2310.18341)** and **LLaVA-Rad (Microsoft)** — CXR-specialist VLMs, fine-tuned
  on radiology data. We deliberately used a **general-purpose, non-fine-tuned** VLM
  (Llama-4-Scout) instead, to test the lower-barrier claim (§6); a specialist model might perform
  better on V4b and is noted as future work.

### 2f. Broader shortcut-mitigation frameworks (methodologically distinct, not competitors)
- **"Preventing Shortcut Learning via Intermediate Layer Knowledge Distillation from Specialist
  Teachers"** (arXiv:2511.17421, FAIMI 2025) and **ShortKit-ML** (medRxiv 2026.04.29.26352053,
  2026) are general medical-imaging shortcut-mitigation/benchmarking frameworks (distillation;
  a unified toolkit respectively), unrelated in mechanism to our detect→name→mitigate pipeline.
  Mentioned for completeness in CO3, not contrasted in detail.

## 3. Dataset and annotation credit

Real chest-tube ground truth for V4b/V5/V6 comes from **NEATX** — *Augmenting Chest X-ray
Datasets with Non-Expert Annotations* (Cheplygina, Damgaard, Eriksen, Juodelyte, Jiménez-Sánchez,
PURRlab, IT University of Copenhagen; MICCAI MIUA 2025; arXiv:2309.02244) — which provides
chest-drain annotations for a subset of NIH ChestX-ray14 and PadChest. This is a **direct data
dependency**, credited here distinctly from the "prior art" framing above: NEATX supplied the
labels our V4b/V5 tube experiments and V6's (ultimately negative) tube-transfer test rely on.

## 4. Baselines and where our spikes stand (final, measured)

| Capability | Prior art | Our status (measured, final) |
|---|---|---|
| Shortcut exists / detectable | Zech, DeGrave, Jiménez-Sánchez et al. (manual/statistical) | **V2 PASS** — induced reliance gap **44pp** (synthetic tag) |
| Localize the shortcut | Grad-CAM | **V3 PASS** — peak-in-region **100%** (control 0.5%) |
| **Name the shortcut, blind, synthetic** | *no medical-VLM equivalent found* | **V4a PASS** — P 0.978 / R 0.90 / hallucination 0.02 |
| **Name the shortcut, blind, real device** | *no medical-VLM equivalent found; HalluCXR corroborates high general hallucination* | **V4b FAIL** — P 0.57 / R 0.66 / hallucination **0.50** (fabricates devices on clean crops) → **documented pivot** |
| Mitigate (no target labels) | GroupDRO / JTT / DFR; concurrent arXiv:2604.17110 (different method, same shortcut) | **V5 PASS (real tube)** — worst-group 0.67→0.83, ID preserved 0.96→0.97; **V5 partial (synthetic tag)** — gap closes but ID drops 0.94→0.62 on a *pure* shortcut model |
| Backbone representation quality under shortcut reliance | Kirichenko et al. follow-up caution (arXiv:2308.00473) | **Linear probe PASS-informative** — frozen p=0.95 backbone probes to AUC **0.690** (chance 0.50, clean 0.845): degraded, not hollow |
| Cross-site OOD gap exists | Zech (hospital-system leakage); Rafferty & Rajan 2026 (NIH/PadChest label-quality issues) | **V6 PASS (gap exists)** — NIH 0.842 → PadChest 0.650, drop **0.192** |
| Cross-site gap is shortcut-driven | *no equivalent test found* | **V6 tested and NOT supported** — zero-shot source-side DFR (no target labels): **+0.005, null**. Supervised target adaptation (+0.082) helps but is not a mitigation claim (see §6) |

## 5. Course-outcome linkage
- **CO1 (this doc):** problem + literature map + novelty, updated against final results.
- **CO2:** auditor architecture/module design (`src/auditor/`).
- **CO3:** the complete spike-driven experiments (V1–V6), repo, and reproducibility.
- **CO4:** evaluation, corrected/honest claims, limitations, cross-hospital risk, fairness,
  patient safety.

## 6. Novelty framing (final, corrected)

Our contribution is an **auditor that names shortcuts prompt-free and generative-model-free**,
localizes them, and tests mitigation — with an honest accounting of where each step succeeds,
fails, or is inconclusive. Positioned against the closest work:

- **vs RoentMod (generative counterfactuals):** RoentMod *edits images with a diffusion pipeline*
  to correct shortcuts. We instead **name** the shortcut from Grad-CAM crops using a
  general-purpose, non-fine-tuned open-weight VLM (Llama-4-Scout), with **no generative model,
  no fine-tuning, and low compute** (~200 crops, one free API pass). Lower barrier, complementary
  goal — and our V4b result shows this naming approach has a real ceiling on hard, realistic
  confounds that RoentMod's correction-only design does not need to clear.
- **vs "From Clinical Intent to Clinical Model" (concurrent, arXiv:2604.17110):** both target the
  real chest-drain/pneumothorax shortcut; we use DFR with an explicit worst-group/ID-preservation
  metric and no target labels, they use an autonomous coding-agent reporting % reliance reduction.
  Independent convergence on the same problem is corroborating, not competing.
- **vs Zech / DeGrave / Jiménez-Sánchez (manual or statistical diagnosis):** they demonstrate
  shortcuts exist by expert analysis; we **attempt to automate the naming** step and gate it on a
  precision-over-recall, hallucination-bounded metric — and report, rather than hide, that
  automation **fails** on the real-device case (V4b), which none of the manual-diagnosis papers
  needed to risk.
- **vs LADDER (closest label-free naming prior art):** both name a classifier's shortcut using a
  foundation model without pre-specified attribute labels, in radiology. LADDER routes the image
  through a **captioner + text-LLM** and evaluates via worst-group accuracy after rectification;
  we route the **saliency crop through a VLM directly** and evaluate the *name itself* with a
  control-hallucination gate — which is what let us **measure and report the naming failure (V4b)**
  that a mitigation-only evaluation would never have to expose.
- **vs SpurLens / RaVL (general-domain VLM auditing):** both are the closest *mechanism*
  analogues (automated, largely prompt-free shortcut discovery via a VLM/LLM), but neither
  touches medical imaging, blind clinical naming, or a control-hallucination-gated metric — the
  gap they leave open in the medical/CXR setting is exactly where this project sits.
- **vs GroupDRO / JTT / DFR (mitigation only):** these assume you already know the groups; our
  loop attempts to *discover and name* the confound first. That said, our V6 zero-shot test shows
  the discover→mitigate chain does **not** demonstrably improve cross-site transfer when the
  named shortcut (chest tube) is reduced at the source — a genuinely novel negative result we
  believe is unpublished elsewhere (§7).

**Corrected scope (two claims walked back after a targeted review):**
1. *"Mitigation validated across three settings"* is **withdrawn**. V5's real-tube result is
   the only clean cross-setting mitigation claim (no target-distribution labels used). V6's
   PadChest recovery (+0.082) is **supervised target-domain adaptation**, which would help even
   absent any shortcut, and does not license a mitigation claim; the honest zero-shot source-DFR
   test that *would* license it returned a null result (+0.005).
2. *"Pure-shortcut model is hollow"* is **withdrawn**. A linear probe on the frozen p=0.95
   backbone reaches AUC 0.690 (chance 0.50, clean baseline 0.845) — the backbone learned real,
   if degraded, pneumothorax features. The correct claim is that shortcut reliance **degrades,
   not eliminates**, core-feature learning, and that this degradation is strength-dependent (the
   correlation sweep, §CO3).

Both corrections came from running the specific control that could have falsified the original
claim (a linear probe; a zero-shot, label-free mitigation test) rather than accepting the more
convenient number. We consider surfacing and correcting these as important as the passing gates.

## 7. What appears genuinely novel (post-search)
No paper found through July 2026 (a) performs blind, general-VLM naming of a chest-X-ray
shortcut scored on precision/recall/control-hallucination, or (b) runs a zero-shot,
label-free source-side intervention explicitly to test whether a known real shortcut explains a
measured cross-site generalization gap, reporting a null result honestly. We believe (a) and (b)
are this project's most defensible individual contributions, ahead of the (correctly humbler)
DFR mitigation results, which have concurrent and prior art.

## References
- Zech et al. (2018), PLOS Medicine — https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- DeGrave, Janizek, Lee (2021), Nature Machine Intelligence — https://www.nature.com/articles/s42256-021-00338-7
- Jiménez-Sánchez, Juodelyte, Chamberlain, Cheplygina (2023) — Detecting Shortcuts in Medical Images
- RoentMod (2025/2026), npj Digital Medicine — https://www.nature.com/articles/s41746-026-02497-6 · arXiv:2509.08640
- Sagawa et al. (2020), GroupDRO — arXiv:1911.08731
- Liu et al. (2021), JTT — arXiv:2107.09044
- Kirichenko et al. (2022), DFR — arXiv:2204.02937 · follow-up arXiv:2308.00473 · arXiv:2409.14637
- Zhao, Hauke, De Castilhos, Bode, Kather, Nebelung, Truhn (2026) — arXiv:2604.17110
- Ghosh, Syed, Wang, Choudhary, Li, Poynton, Visweswaran, Batmanghelich (2025), LADDER — Findings of ACL 2025, arXiv:2408.07832
- Hosseini, Nawathe, Moayeri, Balasubramanian, Feizi (2025), SpurLens — arXiv:2503.08884
- Varma, Delbrouck, Chen, Chaudhari, Langlotz (2024), RaVL — arXiv:2411.04097
- Fay, Reguigui, Yang, Gatidis, Küstner (2025), MIMM-X — arXiv:2511.22990
- Cheplygina, Damgaard, Eriksen, Juodelyte, Jiménez-Sánchez (2025), NEATX — arXiv:2309.02244 · https://zenodo.org/records/14944064
- Sourget, Restrepo, Hudelot, Ferrante, Christodoulidis, Vakalopoulou (2025) — arXiv:2507.21291
- Rafferty & Rajan (2026) — arXiv:2509.15107
- Wang & Li (2026), HalluCXR — arXiv:2605.20469
- Lotfinia, Ziegelmayer, Adams, Truhn, Maier, Arasteh (2026) — arXiv:2606.17710
- CXR-LLaVA — arXiv:2310.18341 · LLaVA-Rad — https://github.com/microsoft/LLaVA-Rad
