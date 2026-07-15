from pathlib import Path
import pytest
from auditor.discovery.key_pool import load_dead_keys
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


def test_daily_quota_429_marks_key_dead_and_skips_it_for_rest_of_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    dead_keys_path = tmp_path / "dead_keys.json"
    calls = []

    daily_quota_payload = {"error": {
        "message": "Rate limit reached for model `x` on tokens per day (TPD): Limit 500000, Used 499999.",
        "type": "tokens", "code": "rate_limit_exceeded"}}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer k1":
            return _FakeResp(429, daily_quota_payload)
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    monkeypatch.setattr("auditor.discovery.providers.time.sleep", lambda s: None)
    prov = OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x",
                                    dead_keys_path=dead_keys_path)

    out1 = prov.describe_crop(_png(tmp_path), "describe")
    assert out1 == "ok"
    assert calls == ["Bearer k1", "Bearer k2"]  # k1 hit the daily-quota 429 once

    calls.clear()
    out2 = prov.describe_crop(_png(tmp_path), "describe")
    assert out2 == "ok"
    assert calls == ["Bearer k2"]  # k1 skipped entirely this time — no wasted attempt

    assert load_dead_keys(dead_keys_path, "fake") == {"k1"}


def test_fresh_provider_loads_persisted_dead_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    dead_keys_path = tmp_path / "dead_keys.json"
    from auditor.discovery.key_pool import record_dead_key
    record_dead_key(dead_keys_path, "fake", "k1")

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    prov = OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x",
                                    dead_keys_path=dead_keys_path)
    prov.describe_crop(_png(tmp_path), "describe")
    assert calls == ["Bearer k2"]  # k1 was already known-dead from a prior process


def test_missing_keys_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x")


def test_openai_provider_backs_off_when_all_keys_429(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    sleeps = []
    monkeypatch.setattr("auditor.discovery.providers.time.sleep", lambda s: sleeps.append(s))
    responses = [_FakeResp(429), _FakeResp(429),
                 _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    prov = OpenAICompatibleProvider("fake", "https://x/v1", "FAKE_KEYS", "model-x")
    out = prov.describe_crop(_png(tmp_path), "describe")
    assert out == "ok"
    # both keys 429'd (one full cycle) before success -> exactly one back-off sleep fired
    assert len(sleeps) == 1


from auditor.discovery.providers import GeminiProvider, build_provider


def test_gemini_provider_parses_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "gk1")
    def fake_post(url, headers=None, json=None, timeout=None, params=None):
        assert "generativelanguage.googleapis.com" in url
        return _FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "a line"}]}}]})
    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    prov = GeminiProvider("GEMINI_API_KEYS", "gemini-2.0-flash")
    assert prov.describe_crop(_png(tmp_path), "describe") == "a line"


def test_build_provider_routes_by_name(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "g1")
    p = build_provider({"provider": "groq", "model": "m"})
    assert p.name == "groq" and p.model == "m"


def test_gemini_empty_parts_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "gk1")
    def fake_post(url, headers=None, json=None, timeout=None, params=None):
        return _FakeResp(200, {"candidates": [{"content": {"parts": []}}]})
    monkeypatch.setattr("auditor.discovery.providers.requests.post", fake_post)
    from auditor.discovery.providers import GeminiProvider
    prov = GeminiProvider("GEMINI_API_KEYS", "gemini-2.0-flash")
    assert prov.describe_crop(_png(tmp_path), "describe") == ""
