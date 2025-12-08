import pytest
import torch
from torch.autograd import gradcheck

from mage.autograd import MageAdd, MageFma, MageRelu


class TestAddBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(100, device=device, requires_grad=True)
        y = torch.randn(100, device=device, requires_grad=True)
        result = MageAdd.apply(x, y)
        result.sum().backward()
        assert x.grad is not None
        assert y.grad is not None

    def test_gradient_value(self, device):
        x = torch.randn(100, device=device, requires_grad=True)
        y = torch.randn(100, device=device, requires_grad=True)
        result = MageAdd.apply(x, y)
        result.sum().backward()
        # d(sum(x+y))/dx = 1 for each element
        assert torch.allclose(x.grad, torch.ones_like(x))
        assert torch.allclose(y.grad, torch.ones_like(y))

    def test_gradcheck(self, device):
        x = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        y = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageAdd.apply, (x, y), eps=1e-6, atol=1e-4)


class TestReluBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(100, device=device, requires_grad=True)
        result = MageRelu.apply(x)
        result.sum().backward()
        assert x.grad is not None

    def test_gradient_value(self, device):
        x = torch.tensor([-1.0, 0.5, -0.5, 1.0], device=device, requires_grad=True)
        result = MageRelu.apply(x)
        result.sum().backward()
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        assert torch.allclose(x.grad, expected)

    def test_matches_torch(self, device):
        x = torch.randn(1000, device=device, requires_grad=True)

        # Triton
        result = MageRelu.apply(x)
        result.sum().backward()
        triton_grad = x.grad.clone()

        # PyTorch reference
        x.grad = None
        ref = torch.relu(x)
        ref.sum().backward()

        assert torch.allclose(triton_grad, x.grad)

    def test_gradcheck(self, device):
        # Avoid values near 0 where gradient is undefined
        x = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        x.data = x.data.abs() + 0.1
        assert gradcheck(MageRelu.apply, (x,), eps=1e-6, atol=1e-4)


class TestFmaBackward:
    def test_gradient_exists(self, device):
        a = torch.randn(100, device=device, requires_grad=True)
        x = torch.randn(100, device=device, requires_grad=True)
        y = torch.randn(100, device=device, requires_grad=True)
        result = MageFma.apply(a, x, y)
        result.sum().backward()
        assert a.grad is not None
        assert x.grad is not None
        assert y.grad is not None

    def test_matches_torch(self, device):
        a = torch.randn(100, device=device, requires_grad=True)
        x = torch.randn(100, device=device, requires_grad=True)
        y = torch.randn(100, device=device, requires_grad=True)

        # Triton
        result = MageFma.apply(a, x, y)
        result.sum().backward()
        triton_grads = (a.grad.clone(), x.grad.clone(), y.grad.clone())

        # PyTorch reference
        a.grad = x.grad = y.grad = None
        ref = a * x + y
        ref.sum().backward()

        assert torch.allclose(triton_grads[0], a.grad)
        assert torch.allclose(triton_grads[1], x.grad)
        assert torch.allclose(triton_grads[2], y.grad)

    def test_gradcheck(self, device):
        a = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        x = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        y = torch.randn(10, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageFma.apply, (a, x, y), eps=1e-6, atol=1e-4)
