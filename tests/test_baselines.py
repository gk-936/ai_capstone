"""Phase 3 baseline tests: JTT upweighting helps a group ERM fails, + table aggregation."""

import numpy as np

from auditor.mitigation.baselines import erm_predictions, fit_jtt_head
from auditor.mitigation.dfr import fit_dfr_head
from auditor.mitigation.run_ablation import _aggregate, _markdown_table


def test_jtt_improves_group_erm_misclassifies():
    """Build features where ERM (biased by a majority) mislabels a minority group; JTT should
    recover it by upweighting the errors."""
    rng = np.random.RandomState(0)
    # Core feature (0) weakly separates classes; a spurious feature (1) is strong but only for the
    # MAJORITY. Minority relies on core only.
    def make(n, y, spurious):
        core = rng.normal(1.0 if y else -1.0, 0.5, size=(n, 1))
        spur = rng.normal(2.0 if spurious else -2.0, 0.3, size=(n, 1))
        return np.hstack([core, spur])

    maj_pos = make(300, 1, spurious=True)    # majority pos: spurious agrees
    maj_neg = make(300, 0, spurious=False)
    min_pos = make(30, 1, spurious=False)    # minority pos: spurious DISAGREES (ERM fails these)
    min_neg = make(30, 0, spurious=True)
    feats = np.vstack([maj_pos, maj_neg, min_pos, min_neg]).astype(np.float32)
    labels = np.array([1] * 300 + [0] * 300 + [1] * 30 + [0] * 30)

    # ERM: standard fit -> leans on the spurious feature.
    erm_w, erm_b = fit_dfr_head(feats, labels, seed=0)  # plain logreg = ERM here
    erm_pred = erm_predictions(feats, erm_w, erm_b)
    minority = np.zeros(len(labels), dtype=bool)
    minority[600:] = True
    erm_min_acc = (erm_pred[minority] == labels[minority]).mean()

    # JTT: upweight ERM's mistakes -> should help the minority.
    jtt_w, jtt_b = fit_jtt_head(feats, labels, erm_w, erm_b, upweight=20.0, seed=0)
    jtt_pred = erm_predictions(feats, jtt_w, jtt_b)
    jtt_min_acc = (jtt_pred[minority] == labels[minority]).mean()

    assert jtt_min_acc >= erm_min_acc  # JTT does not hurt, and generally helps, the failed group


def test_jtt_deterministic():
    rng = np.random.RandomState(1)
    feats = rng.normal(size=(200, 4)).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(int)
    ew, eb = fit_dfr_head(feats, labels, seed=1)
    a = fit_jtt_head(feats, labels, ew, eb, seed=1)
    b = fit_jtt_head(feats, labels, ew, eb, seed=1)
    assert np.allclose(a[0], b[0]) and a[1] == b[1]


def test_aggregate_and_table():
    rows = [
        {"confound": "tube", "method": "ERM", "seed": 1, "worst_group": 0.67, "id": 0.96, "gap": 0.29},
        {"confound": "tube", "method": "ERM", "seed": 2, "worst_group": 0.69, "id": 0.96, "gap": 0.27},
        {"confound": "tube", "method": "DFR", "seed": 1, "worst_group": 0.83, "id": 0.97, "gap": 0.14},
        {"confound": "tube", "method": "DFR", "seed": 2, "worst_group": 0.81, "id": 0.97, "gap": 0.16},
    ]
    agg = _aggregate(rows)
    dfr = next(a for a in agg if a["method"] == "DFR")
    assert dfr["worst_group_mean"] == 0.82
    assert dfr["n_seeds"] == 2
    table = _markdown_table(agg)
    assert "worst-group" in table and "DFR" in table and "ERM" in table
