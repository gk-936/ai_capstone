"""Seed determinism: same seed -> same random draws; different seed -> different."""

import random

from auditor.common.seed import set_seed


def _draw():
    return [random.random() for _ in range(5)]


def test_same_seed_reproduces():
    set_seed(123)
    a = _draw()
    set_seed(123)
    b = _draw()
    assert a == b


def test_different_seed_differs():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_numpy_determinism():
    np = __import__("numpy") if _has_numpy() else None
    if np is None:
        return  # numpy not installed; skip silently
    set_seed(7)
    a = np.random.rand(4).tolist()
    set_seed(7)
    b = np.random.rand(4).tolist()
    assert a == b


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False
