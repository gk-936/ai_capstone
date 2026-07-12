# Phase 2 — Multi-VLM Naming Study (design)

**Claim under test:** Is V4b's naming hallucination (control-hallucination 0.50 on real chest-tube
crops, with Llama-4-Scout) a *Scout-specific quirk* or a *general property* of prompt-free VLM
auditing? And can abstention/confidence or larger crops reduce it? Either outcome is publishable:
general failure = a strong, generalizable warning; a specialist (or an ablation) that clears the
gate = a positive capability result. **Honest guardrail (project-wide): if an ablation does not
help, that is the reported result — no tuning-to-pass.**

## Roster (6 models)
Five general VLMs via free APIs (team-key round-robin per provider), one medical specialist on
Colab (separate track):
1. **Llama-4-Scout** (Groq) — incumbent baseline (V4b already scored).
2. **Llama-4-Maverick** (Groq) — same family, larger → the model-size axis within a family.
3. **Gemini Flash** (Google AI Studio) — distinct lab, generous free tier.
4. **GPT-4o** (GitHub Models) — frontier proprietary anchor reviewers name; low free quota
   (~50/token/month since Jun 2026) → needs several pooled team GitHub tokens.
5. **Pixtral** (Mistral La Plateforme) — distinct European lab. (Qwen-VL is a documented one-line
   swap if preferred.)
6. **CheXagent-2 (3B)** (Colab) — medical specialist; SigLIP + Phi-2, VQA/abnormality-detection
   training matches blind "what is in this crop" better than MAIRA-2's report-generation format.
   (MAIRA-2 7B is the documented alternative.)

## Architecture — generalize the proven Groq round-robin
- `KeyPool` (refactored out of `name_groq.py`): a comma-separated key list per provider, with
  round-robin advance per call, rotate-on-429, network/5xx retry+backoff, and back-off only when
  all keys are exhausted in a cycle. Unit-tested.
- `VLMProvider` base: `describe_crop(image_path, prompt) -> str`. One adapter per provider:
  - `GroqProvider` (OpenAI-compatible; Scout + Maverick by model id)
  - `GeminiProvider` (Google AI Studio REST)
  - `GitHubModelsProvider` (OpenAI-compatible Azure endpoint; GPT-4o)
  - `MistralProvider` (OpenAI-compatible; Pixtral)
  Each adapter owns a `KeyPool` from its env var: `GROQ_API_KEYS`, `GEMINI_API_KEYS`,
  `GITHUB_TOKENS`, `MISTRAL_API_KEYS` (comma-separated, loaded from `.env`).
- **File contract unchanged:** `crops/ + manifest.json → predictions_<model>.jsonl`. The
  specialist plugs in via a Colab notebook that writes the same format. Scoring is provider-blind.

## Data set (identical inputs for a fair comparison)
- Reuse the **existing V4b tube crops + manifest** (100 tube-positive + 100 confound-free control)
  so every model sees byte-identical inputs. The larger-crop ablation re-exports at a bigger size.

## Ablations — on {Scout, best-general} only
- **Abstention/confidence:** a prompt variant instructing the model to name a device *only if
  confident it is artificial and clearly present*, else answer "normal / no device"; optionally
  emit a 0–1 confidence to threshold. Targets control-hallucination (the worst metric).
- **Larger crops:** re-export tube crops at ~128px (or full image with the region highlighted) so
  a thin tube is not cut off. Targets recall.
- Applied to the incumbent (Scout — can we fix the documented failure?) and the best-performing
  general model from the core table (is the approach salvageable at its ceiling?).

## Scoring & guardrail
- Same `names_tube` matcher + same crops for all 6 → one P / R / control-hallucination row per
  model = the headline comparison table.
- **Manual audit** of ~15 raw descriptions per model validates the matcher per model before the
  number is trusted (as in V4a).
- If abstention/larger-crops do not help, report that. No tuning-to-pass.

## Data flow
```
existing V4b crops + manifest  (identical for all models)
  -> per provider: KeyPool round-robin -> predictions_<model>.jsonl
  -> score_naming(names_tube) -> P/R/hallucination row  ->  6-model table
  -> ablations {Scout, best}: abstention prompt / larger crops -> same scoring
  -> CheXagent-2 (Colab) -> predictions_chexagent.jsonl -> same scoring
```

## Tests
- `KeyPool`: round-robin advances per call; 429 rotates; backs off only when all keys exhausted;
  network/5xx retried. (Ported + unit-tested; no live keys in tests.)
- Each provider adapter: request-build and response-parse shape mocked (no network).
- Abstention scoring: a confident "normal" must NOT flag; a hedged sub-threshold device mention
  must NOT flag; a confident device mention flags.

## What runs where / user inputs
- **User:** collect free API keys (self + teammates) for Groq (have), Google AI Studio, GitHub
  tokens, Mistral — several each for round-robin headroom — into `.env`.
- **Local (me):** adapter layer + run the 5 general models + scoring + table.
- **Colab (user):** run the CheXagent-2 notebook (provided).

## Runtime
200 crops × 5 models, throttled/round-robined ≈ minutes-to-~1h each depending on free-tier
limits. CheXagent on Colab: minutes.

## Out of scope (explicitly)
- No fine-tuning of any VLM. No generative image editing. No change to the DFR/mitigation code.
- GroupDRO and other mitigation baselines are unaffected.
