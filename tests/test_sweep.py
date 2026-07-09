"""Sweep table-assembler test (training/DFR themselves are covered by V2/V5 tests)."""

from auditor.mitigation.run_sweep import _markdown


def test_markdown_has_all_points_and_headers():
    rows = [
        {"p": 0.6, "gap_trained": 0.21, "worst_group_before": 0.55, "worst_group_after": 0.72,
         "id_before": 0.90, "id_after": 0.88, "gap_after": 0.03},
        {"p": 0.95, "gap_trained": 0.44, "worst_group_before": 0.50, "worst_group_after": 0.61,
         "id_before": 0.94, "id_after": 0.60, "gap_after": -0.01},
    ]
    md = _markdown(rows)
    assert "reliance gap" in md and "worst-group" in md
    assert "| 0.6 |" in md and "| 0.95 |" in md
    assert md.count("->") == 6  # header has 2 ("before->after" x2) + 2 per data row x 2 rows
