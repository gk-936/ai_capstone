import json
from pathlib import Path
import pytest
from auditor.discovery.providers import OpenAICompatibleProvider


class _FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = {}
    def json(self):
        return self._payload
    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should not raise on 200")


def _png(tmp_path) -> Path:
    p = tmp_path / "c.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)  # minimal bytes; not decoded in the test
    return p


def test_openai_provider_returns_content(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["auth"] = headers["Authorization"]
        captured["model"] = json["model"]
        return _FakeResp(200, {"choices": [{"message": {"content": "  a white tube  "}}]})

    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    prov = OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x")
    out = prov.describe_crop(_png(tmp_path), "describe")
    assert out == "a white tube"
    assert captured["url"] == "https://x/v1/chat/completions"
    assert captured["auth"] == "Bearer k1"
    assert captured["model"] == "model-x"


def test_openai_provider_rotates_key_on_429(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return _FakeResp(429)
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    monkeypatch.setattr("auditor.discovery.providers.time.sleep", lambda s: None)
    prov = OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x")
    out = prov.describe_crop(_png(tmp_path), "describe")
    assert out == "ok"
    assert calls == ["Bearer k1", "Bearer k2"]  # rotated to the 2nd key


def test_missing_keys_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x")
