import pytest
import torch

from mage.matmul import matmul


class TestMatmul:
    def test_square(self, device):
        a = torch.randn(128, 128, device=device, dtype=torch.float16)
        b = torch.randn(128, 128, device=device, dtype=torch.float16)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_rectangular(self, device):
        a = torch.randn(64, 256, device=device, dtype=torch.float16)
        b = torch.randn(256, 128, device=device, dtype=torch.float16)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_non_power_of_two(self, device):
        a = torch.randn(63, 127, device=device, dtype=torch.float16)
        b = torch.randn(127, 65, device=device, dtype=torch.float16)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_large(self, device):
        a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-1, rtol=1e-1)

    def test_output_shape(self, device):
        a = torch.randn(32, 64, device=device, dtype=torch.float16)
        b = torch.randn(64, 128, device=device, dtype=torch.float16)
        result = matmul(a, b)
        assert result.shape == (32, 128)

    def test_output_dtype(self, device):
        a = torch.randn(32, 32, device=device, dtype=torch.float16)
        b = torch.randn(32, 32, device=device, dtype=torch.float16)
        result = matmul(a, b)
        assert result.dtype == torch.float16

    def test_dimension_mismatch_raises(self, device):
        a = torch.randn(32, 64, device=device, dtype=torch.float16)
        b = torch.randn(32, 64, device=device, dtype=torch.float16)
        with pytest.raises(AssertionError):
            matmul(a, b)

    def test_non_contiguous_raises(self, device):
        a = torch.randn(64, 32, device=device, dtype=torch.float16).t()  # Non-contiguous
        b = torch.randn(64, 32, device=device, dtype=torch.float16)
        with pytest.raises(AssertionError):
            matmul(a, b)
