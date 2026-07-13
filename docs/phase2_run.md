# Phase 2 — multi-VLM comparison: run instructions

The **code and data are ready**; the run is gated only on (a) team API keys and (b) a healthy
machine. Everything below is copy-paste turnkey. Nothing here has been *run* yet — no results
exist until you execute these commands.

## 0. Fixed inputs (V4b tube crops — verified present, 2026-07-13)
- crops:    `experiments/v4b_tube/20260708_204951_3894b5/crops`  (200 PNGs)
- manifest: `experiments/v4b_tube/20260708_204951_3894b5/manifest.json`  (100 tube + 100 control)

These are the *same* crops Llama-4-Scout scored in V4b (P 0.57 / R 0.66 / control-halluc 0.50).
The question this run answers: **does Scout's 0.50 control-hallucination generalize across VLMs,
or was it model-specific?**

## 1. Keys — put teammates' + your keys in `.env` (gitignored), comma-separated
```
GROQ_API_KEYS=key1,key2,key3
GEMINI_API_KEYS=key1,key2
GITHUB_TOKENS=tok1,tok2
MISTRAL_API_KEYS=key1,key2
```
Any provider whose key var is empty is **skipped automatically** — a partial roster still runs.

## 2. The 6-model comparison table
```
python -m auditor.discovery compare \
  --roster configs/multi_vlm.yaml \
  --crops experiments/v4b_tube/20260708_204951_3894b5/crops \
  --manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json \
  --out-dir experiments/v4b_tube/multi_vlm_compare
```
Writes `comparison_table.md` (P / R / control-halluc per model) + `comparison.json`.
Resumable: each model's raw predictions are cached, so a rerun skips finished crops.

## 3. Ablations (run on Scout + whichever general model scores best in step 2)
Abstention prompt (hedge-suppressing) — attacks the hallucination rate:
```
python -m auditor.discovery name-multi --roster configs/multi_vlm.yaml --model scout \
  --crops experiments/v4b_tube/20260708_204951_3894b5/crops \
  --manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json \
  --out experiments/v4b_tube/multi_vlm_compare/scout_abstention.jsonl --abstention
python -m auditor.discovery score \
  --manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json \
  --predictions experiments/v4b_tube/multi_vlm_compare/scout_abstention.jsonl \
  --confound tube_abstention
```
(`--confound tube_abstention` uses the hedge-suppressing matcher — scoring these predictions with
plain `--confound tube` would NOT measure the abstention effect.)
Larger-crop ablation: re-export tube crops with `--override discovery.crop_size=128` first, then
re-run step 2 against the larger-crop folder.

## 4. Specialist track — CheXagent-2 (Colab, not this machine)
Zip the crops folder to `crops.zip`, upload it + `manifest.json` to
`src/auditor/discovery/colab/chexagent_naming.ipynb` (T4 GPU), run all cells, download
`predictions_chexagent.jsonl`, then score locally:
```
python -m auditor.discovery score \
  --manifest experiments/v4b_tube/20260708_204951_3894b5/manifest.json \
  --predictions predictions_chexagent.jsonl --confound tube
```

## 5. Before trusting ANY number (non-negotiable — `claims-need-controls`)
Manually read ~15 raw descriptions per model against the crops and confirm the keyword matcher
(`names_tube`) is not over- or under-firing. The precision/recall table is only as honest as the
matcher; the manual audit is what licenses the interpretive claim in the writeup.
