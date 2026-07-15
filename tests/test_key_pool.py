import json
import os
import pytest
from auditor.discovery.key_pool import KeyPool, load_dead_keys, load_keys, record_dead_key


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


# ------------------------------- dead-key skipping -------------------------------------

def test_current_skips_dead_keys():
    kp = KeyPool(["a", "b", "c"])
    kp.mark_dead("a")
    assert kp.current() == "b"


def test_current_wraps_past_dead_keys_at_the_end():
    kp = KeyPool(["a", "b", "c"])
    kp.mark_dead("b")
    kp.mark_dead("c")
    kp.rotate()  # idx=1 -> would land on "b", dead
    assert kp.current() == "a"  # only alive key, wraps around to find it


def test_current_falls_back_to_current_index_when_all_dead():
    kp = KeyPool(["a", "b"])
    kp.mark_dead("a")
    kp.mark_dead("b")
    assert kp.current() in ("a", "b")  # no crash; some key returned


def test_rotate_never_lands_on_a_dead_key():
    # rotate() always advances idx by exactly 1 (not skip-aware), so current() must
    # re-skip dead keys on every call — this asserts the dead key never comes back.
    kp = KeyPool(["a", "b", "c"])
    kp.mark_dead("b")
    seq = []
    for _ in range(6):
        seq.append(kp.current())
        kp.rotate()
    assert "b" not in seq
    assert set(seq) == {"a", "c"}


# ------------------------------- persistent dead-key state ------------------------------

def test_record_and_load_dead_key_same_day(tmp_path):
    path = tmp_path / "dead_keys.json"
    record_dead_key(path, "provX", "keyA", today="2026-07-15")
    assert load_dead_keys(path, "provX", today="2026-07-15") == {"keyA"}


def test_load_dead_keys_expires_on_a_new_day(tmp_path):
    path = tmp_path / "dead_keys.json"
    record_dead_key(path, "provX", "keyA", today="2026-07-15")
    assert load_dead_keys(path, "provX", today="2026-07-16") == set()


def test_dead_keys_scoped_per_provider(tmp_path):
    path = tmp_path / "dead_keys.json"
    record_dead_key(path, "groq", "keyA", today="2026-07-15")
    assert load_dead_keys(path, "mistral", today="2026-07-15") == set()
    assert load_dead_keys(path, "groq", today="2026-07-15") == {"keyA"}


def test_load_dead_keys_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nope.json"
    assert load_dead_keys(path, "groq", today="2026-07-15") == set()
