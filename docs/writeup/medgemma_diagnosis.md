# MedGemma-4b-it 400-empty-strings diagnosis

**Subject:** roster benchmark of `google/medgemma-4b-it` produced 400 blank predictions
(`{"description": ""}`) across V4a + V4b.
**Scope:** read-only static analysis of the harness. Nothing was re-run. A human reviews this
before any Colab/Kaggle re-run.

> **BOLD FLAG — empty output is NOT abstention. MedGemma must NOT be reported as a model
> result (not as a hallucination rate, not as a recall, not as "declined to name the tube")
> until the harness bug below is fixed AND the model is re-run and re-scored. A blank
> `description` here is a harness failure, not the model saying "normal anatomy".**

---

## 1. Which code ran, quoted exactly

MedGemma is dispatched through the **generic ImageTextToText** loader + generate pair in all
three roster notebooks. The roster entry (identical in every notebook) is:

```python
{"name": "medgemma_4b_it", "model_id": "google/medgemma-4b-it", "quantize": True,
 "load": load_generic_itt, "generate": generate_generic_itt},
```

So MedGemma is loaded **4-bit quantized** and generated via `generate_generic_itt`.

### 1a. The loader (`vlm_roster_benchmark.ipynb`, "Generic family" cell; byte-identical in `vlm_roster_benchmark_kaggle.ipynb`; same logic in `vlm_roster_remaining_kaggle.ipynb` worker as `load_generic_itt`)

```python
def make_bnb_config() -> "transformers.BitsAndBytesConfig":
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,   # hardcoded bf16, no T4 gate in the Colab nb
        bnb_4bit_quant_type="nf4",
    )

def load_generic_itt(model_id: str, quantize: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if quantize:
        try:
            kwargs["quantization_config"] = make_bnb_config()   # <-- quantizes WHOLE model,
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)  #   incl. vision tower + projector
            return processor, model
        except Exception as e:
            print(f"  4-bit load failed ({e!r}), falling back to bf16")
            kwargs.pop("quantization_config", None)
    kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return processor, model
```

### 1b. The generate function (`vlm_roster_benchmark.ipynb`; the two Kaggle notebooks are line-for-line equivalent, only renamed `gen_generic_itt` in `remaining_kaggle`)

```python
def generate_generic_itt(processor, model, image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)                                   # <-- LINE A: no dtype= cast
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]   # <-- slicing is CORRECT
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()  # <-- LINE B: "" if 0/all-special new tokens
```

### 1c. The per-image + empty guards that let it pass silently

`run_one_dataset` (Colab) / `run_dataset` (Kaggle worker) increments `n_ok` on any **non-raising**
call — even one that returns `""`:

```python
try:
    desc = generate_fn(proc_or_tok, model, image_path, PROMPT)
    n_ok += 1                        # <-- counts a blank-but-non-raising return as OK
except Exception as e:
    desc = ""
    n_failed += 1
    print(f"  IMAGE FAILED {row['id']}: {e!r}")
f.write(json.dumps({"id": row["id"], "kind": row["kind"], "description": desc}) + "\n")
```

And the `remaining_kaggle` "empty ⇒ FAILED" safeguard only inspects `n_ok`, i.e. the exception
count — not whether descriptions are non-empty:

```python
total_ok = sum(r['n_ok'] for r in results)
status = 'OK' if total_ok > 0 else 'FAILED_INFERENCE'   # <-- 400 blank-but-non-raising rows => total_ok=400 => "OK"
```

**Consequence:** if `generate_generic_itt` returns `""` *without raising*, all three notebooks
write 400 blank rows and report the model **OK** — exactly the "reported as a result" trap.

---

## 2. Bug class + evidence

### Ruled out by the code itself
- **(c) extraction/index bug** — *ruled out.* `new_tokens = output[:, inputs["input_ids"].shape[1]:]`
  correctly slices off the full prompt (text **and** the 256 expanded Gemma-3 image soft-tokens,
  which are part of `input_ids`). This matches the MedGemma card's `generation[0][input_len:]`
  contract. The slice is right; it is not stripping a real answer to "".
- **(a) chat-template *shape*** — *ruled out as the shape.* The
  `{"role":"user","content":[{"type":"image","image":<PIL>}, {"type":"text","text":...}]}` +
  `apply_chat_template(add_generation_prompt=True, tokenize=True, return_dict=True)` form is the
  documented Gemma-3 / MedGemma calling convention.

### The defect (systematic, config-level — matches 400/400)
400/400 identical blanks cannot be model content variation; it is a fixed harness/config fault
firing the same way on every crop. Two coupled defects on the MedGemma path produce it:

1. **The entire multimodal model is 4-bit quantized, including the SigLIP vision tower and the
   multi-modal projector** (`quantize: True` → `make_bnb_config()` with no
   `llm_int8_skip_modules`). MedGemma-4b-it is Gemma-3-based; quantizing its vision stack is the
   documented degenerate-generation failure mode — the image embedding is corrupted, greedy
   decoding emits `<end_of_turn>`/EOS as the very first token, and **0 new tokens** survive the
   slice → `batch_decode(...)[0].strip() == ""` at LINE B, **with no exception raised**.
2. **LINE A drops the dtype cast.** The card does `.to(model.device, dtype=torch.bfloat16)`;
   the harness does `.to(model.device)` only, so `pixel_values` stay **float32** while the model
   runs bf16 (Colab) / fp16 (Kaggle T4 path). Depending on the transformers build this either
   feeds a mismatched/NaN image tensor into the vision tower (reinforcing the immediate-EOS,
   0-new-token outcome above) **or raises a dtype `RuntimeError`**.

**Primary classification: (b) — 0 new tokens (immediate EOS) → empty after decode, no exception.**
This is the reading consistent with the failure being *silently reported as a result*: a raising
path would have tripped `remaining_kaggle`'s `total_ok==0 → FAILED_INFERENCE` guard, whereas the
observed outcome is 400 blanks that slipped through as data.

**Alternative: (d) — swallowed dtype exception.** If the user's transformers build raised on the
float32-pixel_values mismatch instead, every crop hit the `except` branch, `desc=""`, and (in the
Colab notebook, which has no empty-guard) still wrote 400 blanks tagged OK.

**One-line discriminator for the reviewer (read the saved console/`run_log`):**
- Console shows `IMAGE FAILED <id>: <repr>` (Colab: ~400 lines; Kaggle worker: first 3) → it was
  **(d)** the dtype/quantization exception.
- Console shows the timing line (`v4a: 400/400 ok ...`) with **no** `IMAGE FAILED` → it was **(b)**
  degenerate 0-token generation.

Both share one root cause (quantized vision stack + missing input dtype cast) and **one fix**.

### Corroboration from the working reference
`chexagent_naming.ipynb` (the known-good path in this harness) does **not** quantize
(`model.to(torch.bfloat16)`, full precision) and slices the same way
(`tokenizer.decode(output[input_ids.size(1):-1])`). Its success where MedGemma failed points at
the **quantization + missing-dtype-cast** delta, not at the slicing or template shape.

---

## 3. Corrected code to paste in

Two edits. Apply to `load_generic_itt`/`make_bnb_config` **and** `generate_generic_itt`
(and the byte-identical `gen_generic_itt` in the `remaining_kaggle` worker). Do **not** edit the
notebooks blindly — a human should apply and review these.

**Fix 1 — never quantize the vision tower / projector; cast inputs to the compute dtype; treat
0-new-tokens as a failure, not a blank result.**

```python
def make_bnb_config(compute_dtype=torch.bfloat16) -> "transformers.BitsAndBytesConfig":
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        # Keep Gemma-3's vision stack OUT of 4-bit — quantizing it yields immediate-EOS / empty gen.
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
    )
```

```python
def generate_generic_itt(processor, model, image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    # Cast FLOATING inputs (pixel_values) to the model's compute dtype; input_ids stay int.
    # BatchFeature.to(dtype=...) only casts float tensors, so this is safe.
    compute_dtype = getattr(model, "dtype", None) or torch.bfloat16
    inputs = inputs.to(model.device, dtype=compute_dtype)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = output[0, input_len:]

    # 0 real new tokens => degenerate/empty generation, NOT an abstention. Raise so the harness
    # records it as FAILED instead of writing a blank description that scores as a model answer.
    if new_tokens.numel() == 0:
        raise RuntimeError("0 new tokens (immediate EOS) — degenerate generation, not abstention")
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("empty decode after stripping special tokens — degenerate generation")
    return text
```

> If `bnb_4bit_compute_dtype`/`llm_int8_skip_modules` still misbehaves on the T4, the safe
> fallback is to load MedGemma-4b **unquantized in bf16** (4B params ≈ 8 GB, fits the ~15.6 GB
> T4) — i.e. give `medgemma_4b_it` `quantize: False`. That most closely matches the model card
> and the working `chexagent_naming.ipynb` path.

**Fix 2 — make the empty guard actually count non-empty descriptions**, so a blank-but-non-raising
return can never be reported OK again (belt-and-suspenders with the `raise` above):

```python
# in run_one_dataset / run_dataset, count a row only if it produced a non-empty description:
desc = generate_fn(proc_or_tok, model, image_path, PROMPT)
if desc.strip():
    n_ok += 1
else:
    n_failed += 1        # blank == failure, never a silent OK
```

---

## 4. Bottom line

- Bug is **systematic and harness-side** (quantized Gemma-3 vision stack + missing input dtype
  cast), not MedGemma behavior. Slicing and template shape are correct.
- Primary class **(b)** 0-new-token empty generation; **(d)** swallowed dtype exception is the
  variant on builds that raise. Reviewer disambiguates via the `IMAGE FAILED` console signal.
- **The 400 blanks are void.** MedGemma-4b-it currently has **no valid V4a/V4b result**. It must
  be re-run with the fix and re-scored before any precision/recall/hallucination number is
  attributed to it, and before it appears in any roster table.
