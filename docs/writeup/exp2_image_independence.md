# Experiment 2 — Image-independence test

Three conditions per positive crop, scored relative to each model's own baseline
(unmodified-crop) description: (a) occlude the RELEVANT region, (b) occlude an IRRELEVANT
region elsewhere, (c) swap in a different same-kind crop entirely. No new ground truth here —
the question is "did the matcher's verdict change, and in which direction", not P/R.

**Status (2026-07-15): ALL 12 of 12 (dataset x condition x model) cells complete at n=100.**

## Raw results, direction-broken-down (measured, as of 2026-07-15)

A combined "verdict changed" rate conflates two very different outcomes, so every cell is
split by flip direction:
- **pos->neg**: device/marker named at baseline, NOT named after occlusion/swap —
  consistent with real grounding (the model's answer tracked the pixels).
- **neg->pos**: NOT named at baseline, a NEW device/marker named after occlusion/swap —
  still fabrication, just relocated onto different pixels, not evidence of grounding.
- **unchanged**: verdict identical before/after (could be correctly-stable OR an artifact —
  see caveat below for why "unchanged" is NOT automatically "no grounding").

| dataset | condition | model | pos->neg | neg->pos | unchanged | verdict-change-rate | n |
|---|---|---|---|---|---|---|---|
| v4a | relevant | scout | 19 | 5 | 76 | 0.24 | 100 |
| v4a | relevant | pixtral | 25 | 22 | 53 | 0.47 | 100 |
| v4a | irrelevant | scout | 16 | 5 | 79 | 0.21 | 100 |
| v4a | irrelevant | pixtral | 27 | 24 | 49 | 0.51 | 100 |
| v4a | swap | scout | 8 | 8 | 84 | 0.16 | 100 |
| v4a | swap | pixtral | 27 | 23 | 50 | 0.5 | 100 |
| v4b | relevant | scout | 61 | 1 | 38 | 0.62 | 100 |
| v4b | relevant | pixtral | 66 | 5 | 29 | 0.71 | 100 |
| v4b | irrelevant | scout | 30 | 4 | 66 | 0.34 | 100 |
| v4b | irrelevant | pixtral | 13 | 2 | 85 | 0.15 | 100 |
| v4b | swap | scout | 20 | 21 | 59 | 0.41 | 100 |
| v4b | swap | pixtral | 17 | 17 | 66 | 0.34 | 100 |

`text-identical-rate` was 0.0 in every cell (dropped from the table above; kept in the raw
JSON) — the VLMs never reproduce their baseline description word-for-word under any
condition, even "irrelevant" occlusion where the confound is untouched. Not a useful
discriminating signal here.

Reported exactly as measured — no target direction/outcome was assumed going in, and none of
the cells above have been smoothed toward an expected pattern.

## Critical methodology caveat — read before interpreting ANY of the above

**Do not treat V4a's relevant-vs-irrelevant comparison as clean. This is a real
contamination in the experiment's own design, not silently fixed, flagged here explicitly.**

`names_confound` (the V4a matcher) fires on any description combining an object-word
(box/square/rectangle/marker/patch/label/...) with a brightness/artificiality word
(white/bright/artificial/synthetic/opaque/...). The flat gray fill used to occlude the tag is
*itself* routinely described by the VLM as "a gray rectangular object... artificial and not
part of the patient's anatomy... a label or marker" — language that trips the exact same
matcher rule the tag was designed to trigger. Spot-checked directly against the raw text:
76/100 scout descriptions under V4a "relevant" occlusion still flag positive (matches the
`unchanged` count above), and reading them shows the model is now describing the *occlusion
patch itself*, not residual tag content.

Checked whether "irrelevant" shows the same artifact: of its 79/100 unchanged-positive scout
cases, the original predictions.jsonl audit (2026-07-14 session notes) found 49/51 sampled
explicitly mention "white" — i.e. they are correctly still detecting the untouched original
tag, which irrelevant occlusion never touches. That is a genuinely different (and correct)
mechanism from relevant's occlusion-patch artifact, even though the raw unchanged-counts look
superficially similar (76 vs 79).

**V4b's matcher (`names_tube`, requires device-specific vocabulary — tube/drain/catheter/
line/wire) is far less susceptible to this artifact**: only 38/100 scout descriptions stay
unchanged-positive under V4b's "relevant" occlusion (vs V4a's 76/100), and spot-checking those
showed most describe a genuinely still-visible SECOND device outside the occluded region,
not a misread of the gray patch itself — a flat gray rectangle doesn't visually resemble
device-specific vocabulary the way it resembles a generic "artificial marker."

**Implication**: V4a's `relevant` numbers above (pos->neg=19, unchanged=76 for scout)
understate true grounding — a large share of the `unchanged` count is the occlusion fill
itself getting misread as an artificial marker, not the model still perceiving the (removed)
tag. V4b's `relevant` numbers are comparatively clean of this specific bias. **Do not read
"V4a's verdict changed less than V4b's" as "V4a is less grounded than V4b" — the measurement
itself is biased toward `unchanged` for V4a in a way it is not for V4b.**

**Limitation for any future re-run**: use a texture-matched occlusion fill (e.g. local mean
intensity / blurred neighborhood content) instead of flat gray, so the occluder itself doesn't
read as an artificial rectangular object to the VLM.

## Operational note (historical — all cells now complete)

Groq enforces a daily token cap (TPD) per model **per organization**, shared across however
many API keys sit under that org — not a per-key limit. Naming runs across 2026-07-14 into
2026-07-15 repeatedly exhausted this budget across all 6 available keys (spanning multiple
orgs); observed the pool "reset" and re-exhaust in small trickles multiple times within a
single day, not just at one fixed daily boundary. `v4a/swap/scout` was the last holdout,
climbing 11 -> 59 -> 100 across several rounds spanning both days. Built reusable tooling for
this during the session (not one-off scripts), still useful for Exp 3 and beyond:
- `src/auditor/discovery/key_pool.py` — `KeyPool` tracks and skips keys already known
  dead-for-today (detected via the response body's "tokens per day" wording), persisted in
  `.cache/dead_keys.json` (gitignored).
- `src/auditor/discovery/check_keys.py` — a near-free (~12-token text-only, not a
  ~2400-token image call) status check + self-healer: `python -m auditor.discovery.check_keys`.

## Interpretation (2026-07-15) — working, not closed

**"V4b fabrication is pure image-independent prior-emission" — NOT SUPPORTED.** The evidence
is V4b's OWN relevant-vs-irrelevant contrast (same `names_tube` matcher used for both
conditions, so the V4a gray-fill contamination described above does not apply here): occluding
the tube region drops device-naming to pos->neg 61/100 (scout) / 66/100 (pixtral), while
occluding an equal-area irrelevant region drops it far less — 30/100 (scout) / 13/100
(pixtral). Device-naming tracks the tube pixels, not a fixed script recited regardless of
input.

**This conclusion does NOT rest on any V4a-vs-V4b comparison.** Per the contamination caveat
above, V4a's `relevant` cell is contaminated by the occlusion fill and the cross-dataset
comparison is invalid for this purpose. The conclusion above rests only on the within-V4b,
same-matcher, same-condition-pair contrast (relevant vs irrelevant), which is not subject to
that contamination.

**Reconciled mechanism**, read together with the 50-83% control-hallucination rate measured
earlier on V4b/multi-VLM: the VLM appears to carry a strong device-present prior AND ground it
on the salient region — reading a real tube when one is genuinely present, and over-reading
ambiguous normal anatomy as a device when none is present. This is prior-driven over-reading
of an ambiguous salient region, not blind recitation and not clean perception either.

**Supporting, not load-bearing**: V4b's swap condition is close to symmetric (pos->neg roughly
equals neg->pos — 20/21 for scout, 17/17 for pixtral), consistent with the model re-reading
each new swapped-in image rather than pattern-locking onto a fixed answer regardless of input.
This is corroborating detail, not part of the core argument above.

**Status of this interpretation**: working, not a closed result, but no longer blocked on
missing data — `v4a/swap/scout` completed at 100/100 (2026-07-15: pos->neg 8, neg->pos 8,
unchanged 84). As anticipated, finishing it did not change this interpretation, since the
conclusion above never depended on V4a (which sits in the contaminated arm). Proceeding to
Experiment 3 (specialist VLM) next, same protocol, no target outcome assumed.
