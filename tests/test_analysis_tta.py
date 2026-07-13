import pytest


def _toy_model(torch):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1))


def test_bn_adapt_updates_running_stats():
    torch = pytest.importorskip("torch")
    from auditor.analysis.tta import bn_adapt

    model = _toy_model(torch)
    bn = model[1]
    before = bn.running_mean.clone()
    x = torch.randn(32, 4) * 5 + 3  # shifted target distribution
    loader = [(x, torch.zeros(32))]
    bn_adapt(model, loader, device="cpu")
    assert not torch.allclose(before, bn.running_mean)  # stats moved toward target


def test_tent_updates_only_bn_affine():
    torch = pytest.importorskip("torch")
    from auditor.analysis.tta import tent_adapt

    model = _toy_model(torch)
    lin0_w = model[0].weight.detach().clone()
    bn_w = model[1].weight.detach().clone()
    x = torch.randn(32, 4)
    loader = [(x, torch.zeros(32))]
    tent_adapt(model, loader, device="cpu", steps=1, lr=0.1)
    assert torch.allclose(lin0_w, model[0].weight)      # backbone weights frozen
    assert not torch.allclose(bn_w, model[1].weight)    # BN affine updated
