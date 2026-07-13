from auditor.analysis.decompose import build_decomposition


def test_build_decomposition_picks_best_tta_and_bounds_shortcut():
    d, md = build_decomposition(
        ood_drop=0.192,
        site_projection_change=-0.002,           # projecting site out barely moves AUC -> ~0 shortcut
        tta_gains={"bn": 0.03, "tent": 0.05},
        label_note="ptx isolated in 40% of cases; top co-finding chest drains",
    )
    assert d["ood_drop"] == 0.192
    assert d["covariate_recoverable_gain"] == 0.05          # best of bn/tent
    assert d["covariate_recoverable_method"] == "tent"
    assert d["shortcut_projection_change"] == -0.002
    assert "not a clean additive partition" in md.lower()   # honesty caveat present
    assert "0.192" in md
