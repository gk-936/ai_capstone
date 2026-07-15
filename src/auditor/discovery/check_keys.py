"""Cheap key-health check for the Groq (or any OpenAI-compatible) key pool.

Sends ONE minimal TEXT-ONLY request per key (~12 tokens) against the same model the real
naming calls use, and updates .cache/dead_keys.json accordingly: keys that respond 200 are
un-marked (a daily quota can reset mid-day, not just at a fixed boundary — observed
2026-07-14), keys that hit a daily-quota 429 are (re-)marked dead for today.

IMPORTANT: a text-only request is NOT a pure no-op probe — the TPD (tokens-per-day) limit is
scoped per MODEL per ORGANIZATION, and any request to that model (image or not) draws from
the same shared daily pool. A text-only request just avoids the large fixed per-image token
cost (~2400 tokens observed — vision APIs tokenize on a fixed patch grid regardless of actual
image size, so even a 1x1 pixel probe costs almost as much as a real crop). This check still
costs ~12 tokens per key, not zero, but that is negligible next to a 500k daily budget.

Usage:
    python -m auditor.discovery.check_keys                          # groq, scout model
    python -m auditor.discovery.check_keys --provider mistral --env-var MISTRAL_API_KEYS \\
        --model pixtral-12b-2409 --base-url https://api.mistral.ai/v1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from auditor.common.paths import REPO_ROOT
from auditor.discovery.key_pool import load_dead_keys, load_keys, record_dead_key
from auditor.discovery.providers import _DEFAULT_DEAD_KEYS_PATH, _is_daily_quota_error


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def check_keys(env_var: str, provider: str, base_url: str, model: str,
               dead_keys_path: str | Path = _DEFAULT_DEAD_KEYS_PATH) -> list[dict]:
    keys = load_keys(env_var)
    if not keys:
        print(f"no keys found in {env_var}")
        return []

    was_dead = load_dead_keys(dead_keys_path, provider)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],  # text-only: ~12 tokens, not ~2400
        "temperature": 0.0,
        "max_tokens": 1,
    }

    rows = []
    for i, key in enumerate(keys):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json=payload, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as exc:
            rows.append({"index": i, "key": key, "status": "network_error", "detail": str(exc)})
            continue

        if r.status_code == 200:
            rows.append({"index": i, "key": key, "status": "alive"})
        elif r.status_code == 429 and _is_daily_quota_error(r):
            rows.append({"index": i, "key": key, "status": "dead_daily_quota"})
            record_dead_key(dead_keys_path, provider, key)
        else:
            rows.append({"index": i, "key": key, "status": f"other_{r.status_code}",
                        "detail": r.text[:150]})

    # Self-heal: a key previously recorded dead that now succeeds gets un-marked. record_dead_key
    # only adds entries, so rewrite the file's provider section from what we just measured.
    alive_now = {row["key"] for row in rows if row["status"] == "alive"}
    reset_keys = was_dead & alive_now
    if reset_keys:
        import json

        state = {}
        p = Path(dead_keys_path)
        if p.exists():
            state = json.loads(p.read_text(encoding="utf-8"))
        state.setdefault(provider, {})
        for key in reset_keys:
            state[provider].pop(key, None)
        p.write_text(json.dumps(state, indent=2), encoding="utf-8")

    n_alive = sum(1 for r in rows if r["status"] == "alive")
    print(f"{provider}/{model}: {n_alive}/{len(rows)} keys alive")
    for row in rows:
        note = " (RESET since last check)" if row["key"] in reset_keys else ""
        print(f"  key[{row['index']}] {row['status']}{note}")
    return rows


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--env-var", default="GROQ_API_KEYS")
    p.add_argument("--provider", default="groq")
    p.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    p.add_argument("--model", default="meta-llama/llama-4-scout-17b-16e-instruct")
    args = p.parse_args()
    check_keys(args.env_var, args.provider, args.base_url, args.model)


if __name__ == "__main__":
    main()
