import os
import pytest
from auditor.discovery.key_pool import KeyPool, load_keys


def test_round_robin_advances_and_wraps():
    kp = KeyPool(["a", "b"])
    seq = []
    for _ in range(5):
        seq.append(kp.current())
        kp.rotate()
    assert seq == ["a", "b", "a", "b", "a"]


def test_size_and_single_key():
    kp = KeyPool(["only"])
    assert kp.size == 1
    kp.rotate()
    assert kp.current() == "only"


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        KeyPool([])


def test_load_keys_parses_dedups_preserves_order(monkeypatch):
    monkeypatch.setenv("MY_KEYS", "k1, k2 ,k1,k3")
    assert load_keys("MY_KEYS") == ["k1", "k2", "k3"]


def test_load_keys_missing_returns_empty(monkeypatch):
    monkeypatch.delenv("MY_KEYS", raising=False)
    assert load_keys("MY_KEYS") == []
