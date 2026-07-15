from auditor.discovery.check_keys import check_keys
from auditor.discovery.key_pool import load_dead_keys, record_dead_key


class _FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_check_keys_marks_daily_quota_dead(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1,k2")
    dead_path = tmp_path / "dead.json"

    def fake_post(url, headers=None, json=None, timeout=None):
        if headers["Authorization"] == "Bearer k1":
            return _FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})
        return _FakeResp(429, {"error": {"message": "... on tokens per day (TPD) ...",
                                         "type": "tokens", "code": "rate_limit_exceeded"}})

    monkeypatch.setattr("auditor.discovery.check_keys.requests.post", fake_post)
    rows = check_keys("FAKE_KEYS", "fake", "https://x/v1", "model-x", dead_keys_path=dead_path)

    assert rows[0]["status"] == "alive"
    assert rows[1]["status"] == "dead_daily_quota"
    assert load_dead_keys(dead_path, "fake") == {"k2"}


def test_check_keys_self_heals_a_previously_dead_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEYS", "k1")
    dead_path = tmp_path / "dead.json"
    record_dead_key(dead_path, "fake", "k1")
    assert load_dead_keys(dead_path, "fake") == {"k1"}

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("auditor.discovery.check_keys.requests.post", fake_post)
    check_keys("FAKE_KEYS", "fake", "https://x/v1", "model-x", dead_keys_path=dead_path)

    assert load_dead_keys(dead_path, "fake") == set()  # un-marked since it responded 200
