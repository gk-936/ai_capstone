import numpy as np
import pytest


def test_disease_head_arrays_extracts_weight_and_bias():
    torch = pytest.importorskip("torch")
    from auditor.analysis.run_site import disease_head_arrays

    class Dummy:
        pass
    m = Dummy()
    m.classifier = torch.nn.Linear(4, 1)
    with torch.no_grad():
        m.classifier.weight.copy_(torch.arange(4, dtype=torch.float32).view(1, 4))
        m.classifier.bias.copy_(torch.tensor([0.5]))
    w, b = disease_head_arrays(m)
    assert w.shape == (4,)
    assert np.allclose(w, [0, 1, 2, 3])
    assert abs(b - 0.5) < 1e-6
