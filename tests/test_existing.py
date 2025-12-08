import pytest
import torch

from mage import add, fma, relu, softmax


class TestAdd:
    def test_basic(self, device):
        x = torch.randn(1000, device=device)
        y = torch.randn(1000, device=device)
        result = add(x, y)
        expected = x + y
        assert torch.allclose(result, expected)

    def test_broadcast_shapes(self, device):
        for shape in [(100,), (100, 100), (32, 64, 128)]:
            x = torch.randn(shape, device=device)
            y = torch.randn(shape, device=device)
            assert torch.allclose(add(x, y), x + y)


class TestFma:
    def test_basic(self, device):
        a = torch.randn(1000, device=device)
        x = torch.randn(1000, device=device)
        y = torch.randn(1000, device=device)
        result = fma(a, x, y)
        expected = a * x + y
        # Fused operations can have slightly different rounding
        assert torch.allclose(result, expected, atol=1e-6)


class TestRelu:
    def test_basic(self, device):
        x = torch.randn(1000, device=device)
        result = relu(x)
        expected = torch.relu(x)
        assert torch.allclose(result, expected)

    def test_negative_values(self, device):
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
        result = relu(x)
        assert (result[:2] == 0).all()
        assert (result[3:] > 0).all()


class TestSoftmax:
    def test_basic(self, device):
        x = torch.randn(32, 64, device=device)
        result = softmax(x)
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_sums_to_one(self, device):
        x = torch.randn(32, 64, device=device)
        result = softmax(x)
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(32, device=device))
