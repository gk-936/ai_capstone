"""Exp 2 scoring: verdict-change direction must be split, not just counted as one rate —
a positive->negative flip (device named at baseline, gone after occlusion) means something
very different from a negative->positive flip (nothing named, a NEW device appears)."""

import json

from auditor.discovery.score_occlusion import score_condition


def _matcher(text: str) -> bool:
    return "DEVICE" in text


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_direction_breakdown_counts_both_flip_types_separately(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    condition_path = tmp_path / "condition.jsonl"

    _write_jsonl(baseline_path, [
        {"id": "a", "description": "DEVICE present"},   # pos -> neg (a)
        {"id": "b", "description": "DEVICE present"},   # pos -> neg (a)
        {"id": "c", "description": "normal"},            # neg -> pos (b)
        {"id": "d", "description": "normal"},            # unchanged (neg->neg)
        {"id": "e", "description": "DEVICE present"},    # unchanged (pos->pos)
    ])
    _write_jsonl(condition_path, [
        {"id": "a", "description": "normal"},
        {"id": "b", "description": "normal"},
        {"id": "c", "description": "DEVICE present"},
        {"id": "d", "description": "normal"},
        {"id": "e", "description": "DEVICE present"},
    ])

    metrics = score_condition(baseline_path, condition_path, _matcher, ["a", "b", "c", "d", "e"])

    assert metrics["n_scored"] == 5
    assert metrics["pos_to_neg"] == 2   # a, b
    assert metrics["neg_to_pos"] == 1   # c
    assert metrics["unchanged"] == 2    # d, e
    assert metrics["verdict_change_rate"] == round(3 / 5, 4)


def test_direction_breakdown_all_unchanged(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    condition_path = tmp_path / "condition.jsonl"
    _write_jsonl(baseline_path, [{"id": "a", "description": "normal"}])
    _write_jsonl(condition_path, [{"id": "a", "description": "normal"}])

    metrics = score_condition(baseline_path, condition_path, _matcher, ["a"])
    assert metrics["pos_to_neg"] == 0
    assert metrics["neg_to_pos"] == 0
    assert metrics["unchanged"] == 1
    assert metrics["verdict_change_rate"] == 0.0


def test_missing_ids_excluded_from_direction_counts(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    condition_path = tmp_path / "condition.jsonl"
    _write_jsonl(baseline_path, [{"id": "a", "description": "DEVICE present"}])
    _write_jsonl(condition_path, [])  # missing entirely

    metrics = score_condition(baseline_path, condition_path, _matcher, ["a"])
    assert metrics["n_missing"] == 1
    assert metrics["n_scored"] == 0
    assert metrics["pos_to_neg"] == 0
    assert metrics["neg_to_pos"] == 0
    assert metrics["unchanged"] == 0
