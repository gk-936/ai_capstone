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

BLIND_PROMPT = (
    "This is a small region that a diagnostic model focused on within a chest X-ray. "
    "Describe anything artificial, non-anatomical, or out of place that you see in this region. "
    "If everything looks like normal anatomy, say so. Be specific and concise."
)


def _api_keys() -> list[str]:
    """All configured Groq keys, in priority order.

    Reads (in order): GROQ_API_KEYS (comma-separated), GROQ_API_KEY_1/_2/... , then GROQ_API_KEY.
    Multiple keys let the run rotate to an overflow key when the primary hits its rate limit —
    intended for a consented team/research pool, kept to this project.
    """
    import os

    keys: list[str] = []
    csv = os.environ.get("GROQ_API_KEYS", "")
    keys += [k.strip() for k in csv.split(",") if k.strip()]
    i = 1
    while True:
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
    single = os.environ.get("GROQ_API_KEY", "").strip()
    if single:
        keys.append(single)

    # De-dup preserving order.
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    if not out:
        raise RuntimeError(
            "No Groq key set. Get a free key at https://console.groq.com (no card), then set "
            "GROQ_API_KEY (or GROQ_API_KEYS=k1,k2,k3 for a rotation pool) in .env."
        )
    return out


def list_models() -> list[str]:
    """Return available Groq model ids (to pick a current vision model)."""
    key = _api_keys()[0]
    r = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {key}"},
                     timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def _encode(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


# Persistent round-robin pointer into the key pool: advances on EVERY call so the keys share
# load evenly (call i -> keys[i % n]), delaying when any single key nears its limit.
_KEY_IDX = 0


def describe_crop(image_path: Path, model: str, max_retries: int = 6,
                  keys: list[str] | None = None) -> str:
    """Call the Groq vision model on one crop with the blind prompt; return the description.

    Round-robin across the key pool: each attempt uses the next key. Consecutive 429s are
    counted; only once all keys have 429'd in a row do we sleep-and-back-off (all limited).
    """
    global _KEY_IDX
    keys = keys or _api_keys()
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
    consecutive_429 = 0
    for attempt in range(max_retries):
        key = keys[_KEY_IDX % len(keys)]
        _KEY_IDX += 1  # round-robin: advance for the next call regardless of outcome
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            r = requests.post(f"{GROQ_BASE}/chat/completions", headers=headers,
                              json=payload, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Transient network drop (RemoteDisconnected, read timeout) — retry, don't crash.
            wait = min(2 ** attempt * 2, 30)
            print(f"  [net] {type(exc).__name__} on {image_path.name}: retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        if r.status_code == 429:  # rate limited — try the next key; back off only if ALL limited
            consecutive_429 += 1
            if consecutive_429 < len(keys):
                print(f"  [429] a key limited on {image_path.name}; trying next key "
                      f"({consecutive_429}/{len(keys)})")
                continue
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2 ** attempt * 3, 60)
            print(f"  [429] all {len(keys)} key(s) limited on {image_path.name}: "
                  f"backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            consecutive_429 = 0  # reset after the cooldown
            continue
        consecutive_429 = 0  # a non-429 response clears the streak
        if r.status_code >= 500:  # transient server error — retry
            wait = min(2 ** attempt * 2, 30)
            print(f"  [5xx] server {r.status_code} on {image_path.name}: retrying in {wait:.1f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"failed after {max_retries} retries on {image_path.name}")


def _already_done(out_path: Path) -> set[str]:
    """Ids already present in a predictions.jsonl, so a re-run resumes instead of restarting."""
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
             model: str, throttle_s: float = 0.5) -> dict:
    """Run the blind prompt over every crop; append to predictions.jsonl, resuming if it exists.

    Resumable: ids already in out_path are skipped, so a rate-limit crash can be continued by
    re-running the same command. Default throttle spaces calls to respect the free-tier RPM cap.
    """
    crops_dir = Path(crops_dir)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    out_path = Path(out_path)

    keys = _api_keys()
    print(f"using {len(keys)} Groq key(s)" + (" with 429 rotation" if len(keys) > 1 else ""))

    done = _already_done(out_path)
    if done:
        print(f"resuming: {len(done)} predictions already present, skipping those")

    n = len(done)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(manifest):
            cid = row["id"]
            if cid in done:
                continue
            desc = describe_crop(crops_dir / f"{cid}.png", model, keys=keys)
            f.write(json.dumps({"id": cid, "description": desc}) + "\n")
            f.flush()
            n += 1
            if i % 20 == 0:
                print(f"{i}/{len(manifest)}  {cid}: {desc[:70]}")
            if throttle_s:
                time.sleep(throttle_s)
    print(f"DONE -> {n} predictions at {out_path}")
    return {"n_predictions": n, "model": model, "out": str(out_path)}
