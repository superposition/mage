import pytest
import torch
from torch.autograd import gradcheck

from mage.autograd import MageSoftmax


class TestSoftmaxBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        result = MageSoftmax.apply(x)
        result.sum().backward()
        assert x.grad is not None

    def test_matches_torch(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)

        # Triton
        result = MageSoftmax.apply(x)
        result.sum().backward()
        triton_grad = x.grad.clone()

        # PyTorch reference
        x.grad = None
        ref = torch.softmax(x, dim=1)
        ref.sum().backward()

        assert torch.allclose(triton_grad, x.grad, atol=1e-5)

    def test_gradcheck(self, device):
        x = torch.randn(8, 16, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageSoftmax.apply, (x,), eps=1e-6, atol=1e-4)

    def test_numerical_stability(self, device):
        # Large values that would overflow without proper handling
        x = torch.randn(32, 64, device=device) * 100
        x.requires_grad_(True)
        result = MageSoftmax.apply(x)
        result.sum().backward()
        assert not x.grad.isnan().any()
        assert not x.grad.isinf().any()

    def test_gradient_shape(self, device):
        x = torch.randn(16, 128, device=device, requires_grad=True)
        result = MageSoftmax.apply(x)
        result.sum().backward()
        assert x.grad.shape == x.shape
