"""VLM provider adapters behind one interface, each with a KeyPool for team-key round-robin.

OpenAICompatibleProvider serves Groq, GitHub Models, and Mistral (all OpenAI /chat/completions
with image_url content). GeminiProvider (Task 3) handles Google AI Studio's different REST shape.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests

from auditor.common.paths import REPO_ROOT
from auditor.discovery.key_pool import KeyPool, load_dead_keys, load_keys, record_dead_key

_DEFAULT_DEAD_KEYS_PATH = REPO_ROOT / ".cache" / "dead_keys.json"


def _b64(image_path: Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _is_daily_quota_error(response) -> bool:
    """True if a 429 response is a DAILY (not per-minute) quota error — worth marking the
    key dead for the rest of today rather than just rotating past it and retrying later."""
    try:
        message = str(response.json().get("error", {}).get("message", "")).lower()
    except (ValueError, AttributeError):
        return False
    return "tokens per day" in message or "requests per day" in message or " tpd" in message


class VLMProvider:
    name: str

    def describe_crop(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(VLMProvider):
    """Groq / GitHub Models / Mistral — OpenAI chat/completions with a base64 image_url."""

    def __init__(self, name: str, base_url: str, env_var: str, model: str, max_retries: int = 6,
                dead_keys_path: str | Path = _DEFAULT_DEAD_KEYS_PATH):
        keys = load_keys(env_var)
        if not keys:
            raise RuntimeError(f"No keys in {env_var}; set it (comma-separated) in .env")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.pool = KeyPool(keys)
        self.max_retries = max_retries
        self.dead_keys_path = dead_keys_path
        for key in load_dead_keys(dead_keys_path, name):
            self.pool.mark_dead(key)

    def describe_crop(self, image_path: Path, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 160,
        }
        consecutive_429 = 0
        for attempt in range(self.max_retries):
            key = self.pool.current()
            self.pool.rotate()
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers,
                                  json=payload, timeout=60)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if r.status_code == 429:
                if _is_daily_quota_error(r):
                    self.pool.mark_dead(key)
                    record_dead_key(self.dead_keys_path, self.name, key)
                    continue  # a dead key doesn't count toward the "all keys limited" backoff
                consecutive_429 += 1
                if consecutive_429 < self.pool.size:
                    continue
                time.sleep(min(2 ** attempt * 3, 60))
                consecutive_429 = 0
                continue
            consecutive_429 = 0
            if r.status_code >= 500:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"{self.name}: failed after {self.max_retries} retries on {Path(image_path).name}")


_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(VLMProvider):
    """Google AI Studio generateContent (contents/parts + inline_data base64)."""

    def __init__(self, env_var: str, model: str, max_retries: int = 6):
        keys = load_keys(env_var)
        if not keys:
            raise RuntimeError(f"No keys in {env_var}; set it (comma-separated) in .env")
        self.name = "gemini"
        self.model = model
        self.pool = KeyPool(keys)
        self.max_retries = max_retries

    def describe_crop(self, image_path: Path, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": _b64(image_path)}},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 160},
        }
        consecutive_429 = 0
        for attempt in range(self.max_retries):
            key = self.pool.current()
            self.pool.rotate()
            url = f"{_GEMINI_BASE}/{self.model}:generateContent"
            try:
                r = requests.post(url, params={"key": key},
                                  headers={"Content-Type": "application/json"},
                                  json=payload, timeout=60)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if r.status_code == 429:
                consecutive_429 += 1
                if consecutive_429 < self.pool.size:
                    continue
                time.sleep(min(2 ** attempt * 3, 60))
                consecutive_429 = 0
                continue
            consecutive_429 = 0
            if r.status_code >= 500:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            r.raise_for_status()
            cands = r.json().get("candidates", [])
            if not cands:
                return ""
            parts = cands[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return (parts[0].get("text", "") or "").strip()
        raise RuntimeError(f"gemini: failed after {self.max_retries} retries on {Path(image_path).name}")


# base URLs + env vars for the OpenAI-compatible providers
_OPENAI_COMPAT = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEYS"),
    "github": ("https://models.inference.ai.azure.com", "GITHUB_TOKENS"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEYS"),
}


def build_provider(spec: dict) -> VLMProvider:
    """spec = {'provider': 'groq'|'github'|'mistral'|'gemini', 'model': <id>} -> provider."""
    prov = spec["provider"]
    model = spec["model"]
    if prov == "gemini":
        return GeminiProvider("GEMINI_API_KEYS", model)
    if prov in _OPENAI_COMPAT:
        base_url, env_var = _OPENAI_COMPAT[prov]
        return OpenAICompatibleProvider(prov, base_url, env_var, model)
    raise ValueError(f"unknown provider {prov!r}")
