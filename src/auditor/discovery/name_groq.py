"""V4a naming via the Groq API (free, open-weight Llama-Vision) — the live-endpoint path.

Same file contract as the Colab notebook: reads crops/ + manifest.json, sends each crop to a
vision model with the BLIND prompt, writes predictions.jsonl. Only the middle step differs.

The model id is a parameter (Groq's free vision model names change); `list_models()` queries the
live /models endpoint so you can pick a valid one. Requires a free key in GROQ_API_KEY (no card).
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

GROQ_BASE = "https://api.groq.com/openai/v1"

# Never names tag/box/corner/marker — the naming must be blind (CLAUDE.md).
BLIND_PROMPT = (
    "This is a small region that a diagnostic model focused on within a chest X-ray. "
    "Describe anything artificial, non-anatomical, or out of place that you see in this region. "
    "If everything looks like normal anatomy, say so. Be specific and concise."
)


def _api_key() -> str:
    import os

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com (no card), then "
            "set it in your shell or in a .env file (see .env.example)."
        )
    return key


def list_models() -> list[str]:
    """Return available Groq model ids (to pick a current vision model)."""
    r = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {_api_key()}"},
                     timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def _encode(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def describe_crop(image_path: Path, model: str, max_retries: int = 4) -> str:
    """Call the Groq vision model on one crop with the blind prompt; return the description."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": BLIND_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_encode(image_path)}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        r = requests.post(f"{GROQ_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
        if r.status_code == 429:  # rate limited — back off and retry
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"rate-limited after {max_retries} retries on {image_path.name}")


def name_all(crops_dir: str | Path, manifest_path: str | Path, out_path: str | Path,
             model: str, throttle_s: float = 0.0) -> dict:
    """Run the blind prompt over every crop in the manifest; write predictions.jsonl."""
    crops_dir = Path(crops_dir)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    out_path = Path(out_path)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(manifest):
            cid = row["id"]
            desc = describe_crop(crops_dir / f"{cid}.png", model)
            f.write(json.dumps({"id": cid, "description": desc}) + "\n")
            n += 1
            if i % 20 == 0:
                print(f"{i}/{len(manifest)}  {cid}: {desc[:70]}")
            if throttle_s:
                time.sleep(throttle_s)
    print(f"DONE -> {n} predictions at {out_path}")
    return {"n_predictions": n, "model": model, "out": str(out_path)}
