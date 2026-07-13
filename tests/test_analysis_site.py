import numpy as np
from auditor.analysis.site_probe import project_out, site_direction, score_features


def test_project_out_removes_component_and_keeps_shape():
    feats = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    d = np.array([1.0, 0.0, 0.0])
    out = project_out(feats, d)
    assert out.shape == feats.shape
    assert np.allclose(out @ (d / np.linalg.norm(d)), 0.0, atol=1e-6)  # orthogonal to d
    assert np.allclose(out[:, 1:], feats[:, 1:])                       # other dims untouched


def test_site_direction_points_from_b_to_a_and_is_unit():
    a = np.array([[10.0, 0.0], [12.0, 0.0]])   # mean (11, 0)
    b = np.array([[0.0, 0.0], [2.0, 0.0]])     # mean (1, 0)
    d = site_direction(a, b)
    assert np.allclose(np.linalg.norm(d), 1.0)
    assert d[0] > 0.99                          # points along +axis0


def test_score_features_is_linear_head():
    feats = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([1.0, -1.0])
    assert np.allclose(score_features(feats, w, 0.5), np.array([0.5 - 1.0, 0.5 - 1.0]))
