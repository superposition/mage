import pytest
import torch

from mage import add, fma, matmul, relu, softmax


class TestTorchCompile:
    def test_add_compile(self, device):
        @torch.compile
        def fn(x, y):
            return add(x, y)

        x = torch.randn(100, device=device)
        y = torch.randn(100, device=device)
        result = fn(x, y)
        expected = x + y
        assert torch.allclose(result, expected)

    def test_relu_compile(self, device):
        @torch.compile
        def fn(x):
            return relu(x)

        x = torch.randn(100, device=device)
        result = fn(x)
        expected = torch.relu(x)
        assert torch.allclose(result, expected)

    def test_fma_compile(self, device):
        @torch.compile
        def fn(a, x, y):
            return fma(a, x, y)

        a = torch.randn(100, device=device)
        x = torch.randn(100, device=device)
        y = torch.randn(100, device=device)
        result = fn(a, x, y)
        expected = a * x + y
        assert torch.allclose(result, expected)

    def test_softmax_compile(self, device):
        @torch.compile
        def fn(x):
            return softmax(x)

        x = torch.randn(32, 64, device=device)
        result = fn(x)
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_matmul_compile(self, device):
        @torch.compile
        def fn(a, b):
            return matmul(a, b)

        a = torch.randn(32, 64, device=device, dtype=torch.float16)
        b = torch.randn(64, 32, device=device, dtype=torch.float16)
        result = fn(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-2)

    def test_chained_ops_compile(self, device):
        @torch.compile
        def fn(x, y):
            z = add(x, y)
            return relu(z)

        x = torch.randn(100, device=device)
        y = torch.randn(100, device=device)
        result = fn(x, y)
        expected = torch.relu(x + y)
        assert torch.allclose(result, expected)

    def test_compiled_backward(self, device):
        @torch.compile
        def fn(x, y):
            return add(x, y).sum()

        x = torch.randn(100, device=device, requires_grad=True)
        y = torch.randn(100, device=device, requires_grad=True)
        loss = fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None
