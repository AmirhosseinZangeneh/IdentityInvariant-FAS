import torch

from identity_invariant_fas.models.grl import GradientReversalLayer


def test_grl_reverses_and_scales_gradient():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    layer = GradientReversalLayer(0.25)
    layer(x).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.25, -0.25]))
