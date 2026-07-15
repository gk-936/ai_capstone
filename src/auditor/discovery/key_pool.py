"""Round-robin API-key pool, ported from the proven name_groq.py rotation logic.

Shared by every provider so multiple team keys can be rotated to beat a provider's per-key
rate limit (round-robin advance per call; callers rotate again on a 429).
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path


class KeyPool:
    """Round-robin key pool that can be told a key is DEAD for the rest of the process
    (e.g. it hit a daily quota, not a transient per-minute limit) and will skip it in
    `current()` from then on, without needing extra failed attempts to rediscover that."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyPool requires at least one key")
        self._keys = list(keys)
        self._idx = 0
        self._dead: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._keys)

    def mark_dead(self, key: str) -> None:
        self._dead.add(key)

    def current(self) -> str:
        n = len(self._keys)
        for offset in range(n):
            key = self._keys[(self._idx + offset) % n]
            if key not in self._dead:
                return key
        return self._keys[self._idx % n]  # all dead — no good choice, return something

    def rotate(self) -> None:
        self._idx += 1


def _today(today: str | None) -> str:
    return today if today is not None else date.today().isoformat()


def record_dead_key(path: str | Path, provider: str, key: str, today: str | None = None) -> None:
    """Persist that `key` hit a DAILY quota today, so future process invocations (each gets
    a fresh in-memory KeyPool) don't have to rediscover it by burning failed attempts."""
    path = Path(path)
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    day = _today(today)
    provider_state = state.setdefault(provider, {})
    provider_state[key] = day
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_dead_keys(path: str | Path, provider: str, today: str | None = None) -> set[str]:
    """Keys recorded dead for `provider` ON TODAY'S DATE — entries from an earlier day are
    treated as expired (a daily quota resets, so a key dead yesterday may be fine now)."""
    path = Path(path)
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    day = _today(today)
    provider_state = state.get(provider, {})
    return {key for key, recorded_day in provider_state.items() if recorded_day == day}


def load_keys(env_var: str) -> list[str]:
    """Comma-separated keys from an env var; de-duplicated, order preserved."""
    raw = os.environ.get(env_var, "")
    out, seen = [], set()
    for part in raw.split(","):
        k = part.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out
