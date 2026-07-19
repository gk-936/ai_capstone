# Experiment 3 — specialist VLM (CheXagent-2-3B) vs general VLMs

Does a domain-specialized medical VLM move the naming failure boundary, or just shift which
priors it emits? Same crops, same blind prompt (byte-verified against `BLIND_PROMPT` in
`name_groq.py`), same file contract as scout/pixtral. No target outcome assumed.

## Raw results — BOTH matcher versions, side by side (measured 2026-07-15)

CheXagent did not answer in free text the way scout/pixtral did. On a large share of crops
it used its own referring-expression/grounding format instead:
`<|ref|> foreign object <|/ref|> <|box|> (20,1),(35,12) <|/box|>` rather than descriptive
prose. The original prose matchers (`names_confound`/`names_tube`) were built to find an
OBJECT-word co-occurring with a BRIGHT/artificiality-word in a sentence, and do not parse
this format — confirmed empirically (see Methodology below), not assumed. Reporting **both**
the old (now known invalid for this model) and new (grounding-aware) numbers, clearly
labeled, per instruction — not silently replacing one with the other.

| Arm | Matcher | Precision | Recall | Control-hallucination | Gate (same bars as V4a/V4b) |
|---|---|---|---|---|---|
| V4a (CheXagent) | OLD: `names_confound` (prose) | 0.50 | 0.01 | 0.01 | FAIL |
| V4a (CheXagent) | NEW: `names_confound_grounding` | **0.9663** | **0.86** | **0.03** | FAIL (R just under 0.90) |
| V4b (CheXagent) | OLD: `names_tube` (prose) | 0.4333 | 0.13 | 0.17 | FAIL |
| V4b (CheXagent) | NEW: `names_tube_grounding` | **0.0** | **0.0** | **0.0** | FAIL |

For reference, the general-VLM numbers already on record for the same crops/prompt:

| Arm | Model | Precision | Recall | Control-hallucination |
|---|---|---|---|---|
| V4a | scout | 0.978 | 0.90 | 0.02 |
| V4a | pixtral | 0.626 | 0.52 | 0.31 |
| V4b | scout | 0.545 | 0.66 | 0.55 |
| V4b | pixtral | 0.494 | 0.81 | 0.83 |

No interpretation of this comparison is drawn here — reported for reference only, exactly as
instructed (stop before interpreting).

## Methodology — why two matchers, and what "grounding-aware" actually checks

**Format mismatch, confirmed not assumed.** Cross-tabbed CheXagent's raw output against
ground truth before writing any matcher: **100/100** V4a positives and **100/100** V4b
positives produced *some* grounded/flagged output (a `<|ref|>` segment, or — for V4a's
free-text branch — anything other than the fixed boilerplate `"No artificial or non-anatomic
structures detected."`). The OLD prose matchers only caught 1/100 (V4a) and 13/100 (V4b) of
those — a stark gap between "the model engaged with the crop" and "the regex recognized it,"
diagnostic of a format problem, not a perception problem. Verified the blocker is structural,
not a missing keyword: even patching the OBJECT word-list to include "object"/"foreign body"
left V4a recall at 1/100, because `"foreign object"` never co-occurs with a BRIGHT-list word
(white/bright/artificial/synthetic/...) the way prose does — CheXagent's grounding labels are
bare categorical class names, not descriptive sentences.

**CheXagent's full grounding label vocabulary** (every unique `<|ref|>` label observed, no
label omitted): V4a = `{foreign object (174), diaphragm (66), other lesion (12), artificial
object (4)}`. V4b = `{diaphragm (307), foreign object (59), other lesion (14), aortic
enlargement (5), artificial object (4), mediastinal (1)}`. **This taxonomy has no
device-specific class at all** — no label anywhere contains tube/catheter/line/drain/wire
vocabulary, in either dataset.

**`names_confound_grounding` (V4a)** routes each crop by its own format: a description
containing `<|ref|>` tokens is checked against a generic label set (`foreign object`,
`foreign body`, `artificial marker/object/structure/device`, `non-anatomical
structure/object`, `artefact`) — matching `names_confound`'s own genericness (no requirement
the object specifically be "tag"/"marker"-shaped, just object+artificiality framing). A
description with no `<|ref|>` tokens (V4a had free-text responses on 71/200 crops — a real
per-crop mix, not one format for the whole arm) falls through unchanged to the original prose
`names_confound`.

**`names_tube_grounding` (V4b)** requires the `<|ref|>` label itself to contain
DEVICE-SPECIFIC vocabulary (tube/catheter/line/drain/wire/cannula/pigtail/thoracostomy/lead),
mirroring `names_tube`'s own specificity. **A bare `"foreign object"` label does NOT count** —
by design, confirmed by a dedicated test
(`test_grounding_v4b_requires_device_specific_vocabulary_not_bare_foreign_object`). This is
why V4b-grounding scores exactly 0/0/0: CheXagent's taxonomy structurally cannot produce a
device-specific label for anything in this dataset, so nothing can pass this bar — **not**
"CheXagent never detects the tube region" (it produced grounding output on 100/100 V4b
positives; the crosstab confirms engagement), but "CheXagent's output vocabulary has no way
to say 'tube' specifically, only a generic 'foreign object' bucket, which was deliberately
excluded from counting as correct so CheXagent isn't held to an easier bar than scout/pixtral
cleared on the same dataset."

Implementation: `names_confound_grounding`/`names_tube_grounding` in
`src/auditor/discovery/score_naming.py`, wired into the existing `score` CLI via
`--confound tag_grounding`/`tube_grounding`. `names_confound`/`names_tube` themselves are
**untouched** — the functions that gated the original V4a/V4b PASS/FAIL decisions are not
modified by this addition. 20 tests added in `tests/test_discovery.py` covering both label
sets, per-crop format routing, multi-segment `<|ref|>` handling, and the empty/None cases.

## Additional VLMs — chexagent-8b, qwen2.5-vl-3b/7b, phi3.5-vision (measured 2026-07-18)

Four more models were run on the identical V4a/V4b crops and blind prompt and scored with the
**existing** project matchers only (no hand-scoring, no new matcher code). Manifests: V4a =
`experiments/v4a_naming/20260708_105244_da01a4/manifest.json` (100 positive + 100 control),
V4b = `experiments/v4b_tube/20260708_204951_3894b5/manifest.json` (100 positive + 100
control). Predictions live under `experiments/exp3_specialist/predictions_<model>_v4{a,b}.jsonl`.

**Which matcher is fair per model — determined by inspecting the raw predictions, not
assumed.** Unlike CheXagent-2-3B, none of these four models emit CheXagent's
`<|ref|>...<|/ref|>` grounding tokens (verified: 0 `<|ref|>` occurrences across all eight
files). They answer (or fail to answer) in free-text prose. Because `names_confound_grounding`
/`names_tube_grounding` route any description **without** `<|ref|>` tokens straight through to
the unmodified prose matcher (`names_confound`/`names_tube`), the grounding matcher and the
prose matcher return **identical numbers** for every one of these models. The **prose matcher
is therefore the fair/primary one** for all four; the grounding column is reported per
instruction and is identical by construction.

Per-model raw output character (from inspecting the predictions):
- **chexagent-8b** — free-text prose, radiology-report style ("No acute cardiopulmonary
  process is observed."). On a share of crops it echoes the prompt back verbatim instead of
  answering. Not grounding format.
- **qwen2.5-vl-3b** — free-text prose, but **degenerate on V4a**: 164/200 responses are the
  refusal "I'm sorry, but I cannot provide an answer as there is no image attached…" (the
  image never reached the model). V4b responses are prose but near-constant boilerplate
  ("no apparent artificial structures…"). Not grounding format.
- **qwen2.5-vl-7b** — free-text prose, genuinely engages with the crop and names
  artificial/foreign objects. Not grounding format.
- **phi3.5-vision** — **empty string for all 200 crops in both arms** (200/200 blank
  `"description": ""`). No usable output produced; every metric is 0 by construction. Not a
  matcher issue — the file contains no text to score.

| Model | Arm | Matcher | Precision | Recall | Control-halluc | Gate | n_scored (missing) |
|---|---|---|---|---|---|---|---|
| chexagent-8b | V4a | tag (prose, fair) | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| chexagent-8b | V4a | tag_grounding | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| chexagent-8b | V4b | tube (prose, fair) | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| chexagent-8b | V4b | tube_grounding | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| qwen2.5-vl-3b | V4a | tag (prose, fair) | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| qwen2.5-vl-3b | V4a | tag_grounding | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| qwen2.5-vl-3b | V4b | tube (prose, fair) | 0.0 | 0.0 | 0.01 | FAIL | 200 (0) |
| qwen2.5-vl-3b | V4b | tube_grounding | 0.0 | 0.0 | 0.01 | FAIL | 200 (0) |
| qwen2.5-vl-7b | V4a | tag (prose, fair) | 0.6346 | 0.33 | 0.19 | FAIL | 200 (0) |
| qwen2.5-vl-7b | V4a | tag_grounding | 0.6346 | 0.33 | 0.19 | FAIL | 200 (0) |
| qwen2.5-vl-7b | V4b | tube (prose, fair) | 0.50 | 0.74 | 0.74 | FAIL | 200 (0) |
| qwen2.5-vl-7b | V4b | tube_grounding | 0.50 | 0.74 | 0.74 | FAIL | 200 (0) |
| phi3.5-vision | V4a | tag (prose, fair) | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| phi3.5-vision | V4a | tag_grounding | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| phi3.5-vision | V4b | tube (prose, fair) | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |
| phi3.5-vision | V4b | tube_grounding | 0.0 | 0.0 | 0.0 | FAIL | 200 (0) |

Notes on the zeros, so they are not misread as identical failures of the same kind:
- **chexagent-8b 0/0/0** — the model produced prose (radiology-report sentences and
  prompt-echoes), but that prose never uses the marker/device vocabulary the matchers key on,
  so recall is 0 with no false positives on controls either.
- **qwen2.5-vl-3b 0/0** — V4a is a pipeline-level non-answer (image not delivered; 164/200
  refusals), so the number reflects a broken run, not the model's perception. V4b is a
  near-constant "nothing artificial here" boilerplate → 0 recall, 1/100 control false positive.
- **phi3.5-vision 0/0/0** — empty output for every crop; nothing was scored beyond blanks. This
  is a missing-output condition, not a matcher-vs-format mismatch.
- **qwen2.5-vl-7b** is the only new model with non-trivial signal (V4a P=0.63/R=0.33, V4b
  R=0.74) but fails every gate, with very high control-hallucination on V4b (0.74).

All sixteen runs used `n_missing_predictions = 0` (every manifest id had a prediction row).
No interpretation of specialist-vs-general is drawn here — numbers reported exactly as the
scorer emitted them.

## Status

Reported exactly as measured — no interpretation drawn on what these numbers mean for the
specialist-vs-general question. That's the next step, pending review.
