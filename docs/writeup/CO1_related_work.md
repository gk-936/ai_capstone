# CO1 — Problem, Related Work, and Novelty Framing

*Spurious-Correlation Auditor for Chest X-ray Classifiers.* This document maps the problem to
prior art and states where our contribution sits. It is written against **measured** results
from our validation spikes (V1–V4a), not aspirations.

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
a full detect→name→mitigate→verify loop.

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

*These motivate the project but do not name/mitigate shortcuts automatically — they diagnose,
by hand, that shortcuts exist.*

### 2b. Correcting shortcuts by generative counterfactuals (the closest prior art)
- **RoentMod, 2025 (npj Digital Medicine; arXiv:2509.08640)** — *A synthetic CXR modification
  model to identify and correct image-interpretation shortcuts.* Combines a medical image
  generator (**RoentGen**) with an image-to-image editor to produce **counterfactual CXRs** with
  user-specified findings while preserving anatomy; training on these counterfactuals improved
  discrimination by **3–19% AUC internally, 1–11% externally**. Reader studies rated 93% of
  images realistic. ([npj Digital Medicine](https://www.nature.com/articles/s41746-026-02497-6),
  [arXiv](https://arxiv.org/abs/2509.08640))

*This is the method we most directly contrast against (see §4): it is powerful but relies on a
**generative diffusion pipeline** (RoentGen + editing) — heavy compute, model fine-tuning, and a
generative apparatus we deliberately avoid.*

### 2c. Mitigation baselines (worst-group robustness)
- **GroupDRO (Sagawa et al., 2020)** — distributionally-robust optimization over predefined
  groups; optimizes worst-group loss. Requires group labels at train time.
- **JTT — Just Train Twice (Liu et al., 2021)** — upweight examples an ERM model gets wrong;
  approximates group robustness without explicit group labels.
- **DFR — Deep Feature Reweighting (Kirichenko et al., 2022)** — retrain **only the last layer**
  on a group-balanced set. Reaches **92.9% worst-group on Waterbirds, 70.1% on CivilComments**,
  matching/beating GroupDRO at a fraction of the cost; premise is that the backbone already
  encodes core features and only their weighting is wrong.
  ([arXiv:2204.02937](https://www.alphaxiv.org/overview/2204.02937v2)) Later work questions
  whether last-layer retraining is always sufficient ([arXiv:2308.00473](https://arxiv.org/abs/2308.00473)).

*These are our V5 mitigation candidates (masking ↔ DFR ↔ unlearning). DFR is the natural first
choice on this hardware: cheap, last-layer-only, strong worst-group gains.*

### 2d. Interpretability / VLM-based auditing
- Grad-CAM-style attribution (Selvaraju et al.) localizes *where* a model looks — our V3 uses it —
  but does not **name** what it found. Recent lines (e.g. CoBalT, MIMM-X, and VLM-as-auditor work)
  push toward describing failure modes; the open gap our project targets is **naming a shortcut
  in clinical language, blind, from attribution crops alone**, with a false-alarm-averse metric.

## 3. Baselines and where our spikes stand (measured)

| Capability | Prior art | Our status (measured) |
|---|---|---|
| Shortcut exists / detectable | Zech, DeGrave (manual) | **V2 PASS** — induced reliance gap **44pp** |
| Localize the shortcut | Grad-CAM | **V3 PASS** — peak-in-region **100%** (control 0.5%) |
| **Name the shortcut, blind** | *sparse; RoentMod corrects but does not name via general VLM* | **V4a PASS** — P 0.978 / R 0.90 / hallucination 0.02 |
| Mitigate | GroupDRO / JTT / DFR | V5 — planned (DFR first) |
| Prove OOD recovery | RoentMod (external AUC gains) | V6 — planned |

## 4. Novelty framing

Our contribution is an **auditor that names shortcuts prompt-free and generative-model-free**,
then mitigates and verifies. Positioned against the closest work:

- **vs RoentMod (generative counterfactuals):** RoentMod *edits images with a diffusion pipeline*
  to correct shortcuts. We instead **name** the shortcut from Grad-CAM crops using a
  general-purpose open-weight VLM (Llama-4-Scout), with **no generative model, no fine-tuning,
  and low compute** (the naming step runs as a free API call over ~200 small crops). Lower
  barrier, complementary goal (naming/auditing vs counterfactual correction).
- **vs Zech / DeGrave (manual diagnosis):** they demonstrate shortcuts exist by expert analysis;
  we **automate the naming** step and gate it on a precision-over-recall, hallucination-bounded
  metric (a fabricated confound costs more than a missed one — an auditor that invents shortcuts
  is worse than none).
- **vs GroupDRO / JTT / DFR (mitigation only):** these assume you already know the groups; our
  loop *discovers and names* the confound first, then feeds mitigation — closing the gap between
  "a shortcut exists" and "here is the named shortcut to mitigate."

**Honest scope (carried from the measured results):** V4a passed on a *synthetic, unambiguous*
confound with recall exactly at the 0.90 bar — 10 genuine VLM misses on an easy target. This
foreshadows that **V4b (real device confound, e.g. chest tube) is materially harder**, and is the
project's documented critical-gate/pivot point. The contribution's strength is the *controlled,
gated demonstration* of prompt-free naming; its limitation is naming reliability on hard,
realistic confounds, which V4b will measure.

## 5. Course-outcome linkage
- **CO1 (this doc):** problem + literature map + novelty. **CO2:** auditor architecture/module
  design. **CO3:** the spike-driven experiments + repo (V1–V4a measured; V5–V6 next). **CO4:**
  evaluation, limitations (naming reliability), cross-hospital risk, fairness, patient safety.

## References
- Zech et al. (2018), PLOS Medicine — https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- DeGrave, Janizek, Lee (2021), Nature Machine Intelligence — https://www.nature.com/articles/s42256-021-00338-7
- RoentMod (2025), npj Digital Medicine — https://www.nature.com/articles/s41746-026-02497-6 · arXiv:2509.08640
- Sagawa et al. (2020), GroupDRO — arXiv:1911.08731
- Liu et al. (2021), JTT — arXiv:2107.09044
- Kirichenko et al. (2022), DFR — arXiv:2204.02937 · follow-up arXiv:2308.00473
