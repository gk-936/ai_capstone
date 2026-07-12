"""VLM provider adapters behind one interface, each with a KeyPool for team-key round-robin.

OpenAICompatibleProvider serves Groq, GitHub Models, and Mistral (all OpenAI /chat/completions
with image_url content). GeminiProvider (Task 3) handles Google AI Studio's different REST shape.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests

from auditor.discovery.key_pool import KeyPool, load_keys


def _b64(image_path: Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


class VLMProvider:
    name: str

    def describe_crop(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(VLMProvider):
    """Groq / GitHub Models / Mistral — OpenAI chat/completions with a base64 image_url."""

    def __init__(self, name: str, base_url: str, env_var: str, model: str, max_retries: int = 6):
        keys = load_keys(env_var)
        if not keys:
            raise RuntimeError(f"No keys in {env_var}; set it (comma-separated) in .env")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.pool = KeyPool(keys)
        self.max_retries = max_retries

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
