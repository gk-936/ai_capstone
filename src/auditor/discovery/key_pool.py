"""Round-robin API-key pool, ported from the proven name_groq.py rotation logic.

Shared by every provider so multiple team keys can be rotated to beat a provider's per-key
rate limit (round-robin advance per call; callers rotate again on a 429).
"""

from __future__ import annotations

import os


class KeyPool:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyPool requires at least one key")
        self._keys = list(keys)
        self._idx = 0

    @property
    def size(self) -> int:
        return len(self._keys)

    def current(self) -> str:
        return self._keys[self._idx % len(self._keys)]

    def rotate(self) -> None:
        self._idx += 1


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
