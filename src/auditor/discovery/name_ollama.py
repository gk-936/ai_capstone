"""V4a/V4b naming via a LOCAL Ollama vision model — no API key, no rate limit.

Same file contract as the Groq path (crops/ + manifest.json -> predictions.jsonl) and the SAME
blind prompt, so scoring is unchanged and V4a can be re-run for a consistent single-model story.
Talks to Ollama's REST API at http://localhost:11434/api/chat with base64 images.

Model is a parameter (any vision tag that fits your VRAM, e.g. qwen2.5vl:3b, qwen3-vl:4b).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from auditor.discovery.name_groq import BLIND_PROMPT  # reuse the exact blind prompt

OLLAMA_BASE = "http://localhost:11434"


def _encode(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def describe_crop(image_path: Path, model: str, host: str = OLLAMA_BASE) -> str:
    """Send one crop to a local Ollama vision model with the blind prompt; return the text."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": BLIND_PROMPT,
            "images": [_encode(image_path)],
        }],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def name_all(crops_dir: str | Path, manifest_path: str | Path, out_path: str | Path,
             model: str, host: str = OLLAMA_BASE) -> dict:
    """Run the blind prompt over every crop via local Ollama; resume-safe, flush per row."""
    crops_dir = Path(crops_dir)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    out_path = Path(out_path)

    done = _already_done(out_path)
    if done:
        print(f"resuming: {len(done)} predictions already present, skipping those")

    n = len(done)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(manifest):
            cid = row["id"]
            if cid in done:
                continue
            desc = describe_crop(crops_dir / f"{cid}.png", model, host)
            f.write(json.dumps({"id": cid, "description": desc}) + "\n")
            f.flush()
            n += 1
            if i % 20 == 0:
                print(f"{i}/{len(manifest)}  {cid}: {desc[:70]}")
    print(f"DONE -> {n} predictions at {out_path}")
    return {"n_predictions": n, "model": model, "out": str(out_path)}
