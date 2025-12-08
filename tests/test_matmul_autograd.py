import pytest
import torch
from torch.autograd import gradcheck

from mage.autograd import MageMatmul


class TestMatmulBackward:
    def test_gradient_exists(self, device):
        a = torch.randn(32, 64, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(64, 32, device=device, dtype=torch.float16, requires_grad=True)
        result = MageMatmul.apply(a, b)
        result.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_gradient_shapes(self, device):
        a = torch.randn(32, 64, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(64, 128, device=device, dtype=torch.float16, requires_grad=True)
        result = MageMatmul.apply(a, b)
        result.sum().backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matches_torch(self, device):
        a = torch.randn(32, 64, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(64, 32, device=device, dtype=torch.float32, requires_grad=True)

        # Triton
        result = MageMatmul.apply(a, b)
        result.sum().backward()
        triton_grad_a = a.grad.clone()
        triton_grad_b = b.grad.clone()

        # PyTorch reference
        a.grad = b.grad = None
        ref = torch.matmul(a, b)
        ref.sum().backward()

        assert torch.allclose(triton_grad_a, a.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(triton_grad_b, b.grad, atol=1e-2, rtol=1e-2)

    @pytest.mark.skip(reason="Matmul kernel uses fp32 accumulators, doesn't support fp64 for gradcheck")
    def test_gradcheck(self, device):
        # Use smaller size and float64 for numerical gradient check
        a = torch.randn(8, 16, dtype=torch.float64, device=device, requires_grad=True)
        b = torch.randn(16, 8, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageMatmul.apply, (a, b), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_rectangular_backward(self, device):
        a = torch.randn(16, 32, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(32, 64, device=device, dtype=torch.float32, requires_grad=True)
        result = MageMatmul.apply(a, b)
        result.sum().backward()

        # Verify shapes
        assert a.grad.shape == (16, 32)
        assert b.grad.shape == (32, 64)
