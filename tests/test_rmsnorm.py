import pytest
import torch
from torch.autograd import gradcheck

from mage import rmsnorm
from mage.normalization import MageRMSNorm


def torch_rmsnorm(x, weight, eps=1e-6):
    """Reference implementation of RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


class TestRMSNormForward:
    def test_basic(self, device):
        x = torch.randn(32, 64, device=device)
        weight = torch.randn(64, device=device)
        result = rmsnorm(x, weight)
        expected = torch_rmsnorm(x, weight)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d_input(self, device):
        x = torch.randn(4, 32, 128, device=device)
        weight = torch.randn(128, device=device)
        result = rmsnorm(x, weight)
        expected = torch_rmsnorm(x, weight)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_output_shape(self, device):
        x = torch.randn(16, 32, 64, device=device)
        weight = torch.randn(64, device=device)
        result = rmsnorm(x, weight)
        assert result.shape == x.shape

    def test_different_eps(self, device):
        x = torch.randn(32, 64, device=device)
        weight = torch.randn(64, device=device)
        for eps in [1e-5, 1e-6, 1e-8]:
            result = rmsnorm(x, weight, eps=eps)
            expected = torch_rmsnorm(x, weight, eps=eps)
            assert torch.allclose(result, expected, atol=1e-5)

    def test_numerical_stability(self, device):
        # Large values that could cause overflow
        x = torch.randn(32, 64, device=device) * 100
        weight = torch.randn(64, device=device)
        result = rmsnorm(x, weight)
        assert not result.isnan().any()
        assert not result.isinf().any()

    def test_small_values(self, device):
        # Small values near zero
        x = torch.randn(32, 64, device=device) * 1e-4
        weight = torch.randn(64, device=device)
        result = rmsnorm(x, weight)
        expected = torch_rmsnorm(x, weight)
        assert torch.allclose(result, expected, atol=1e-5)


class TestRMSNormBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        weight = torch.randn(64, device=device, requires_grad=True)
        result = rmsnorm(x, weight)
        result.sum().backward()
        assert x.grad is not None
        assert weight.grad is not None

    def test_gradient_shapes(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        weight = torch.randn(64, device=device, requires_grad=True)
        result = rmsnorm(x, weight)
        result.sum().backward()
        assert x.grad.shape == x.shape
        assert weight.grad.shape == weight.shape

    def test_matches_torch_grad_x(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        weight = torch.randn(64, device=device, requires_grad=True)

        # Triton
        result = rmsnorm(x, weight)
        result.sum().backward()
        triton_grad_x = x.grad.clone()

        # PyTorch reference
        x.grad = None
        weight.grad = None
        ref = torch_rmsnorm(x, weight)
        ref.sum().backward()

        assert torch.allclose(triton_grad_x, x.grad, atol=1e-4, rtol=1e-4)

    def test_matches_torch_grad_weight(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        weight = torch.randn(64, device=device, requires_grad=True)

        # Triton
        result = rmsnorm(x, weight)
        result.sum().backward()
        triton_grad_weight = weight.grad.clone()

        # PyTorch reference
        x.grad = None
        weight.grad = None
        ref = torch_rmsnorm(x, weight)
        ref.sum().backward()

        assert torch.allclose(triton_grad_weight, weight.grad, atol=1e-4, rtol=1e-4)

    def test_3d_backward(self, device):
        x = torch.randn(4, 32, 128, device=device, requires_grad=True)
        weight = torch.randn(128, device=device, requires_grad=True)

        # Triton
        result = rmsnorm(x, weight)
        result.sum().backward()
        triton_grad_x = x.grad.clone()
        triton_grad_weight = weight.grad.clone()

        # PyTorch reference
        x.grad = None
        weight.grad = None
        ref = torch_rmsnorm(x, weight)
        ref.sum().backward()

        assert torch.allclose(triton_grad_x, x.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(triton_grad_weight, weight.grad, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        x = torch.randn(8, 16, dtype=torch.float64, device=device, requires_grad=True)
        weight = torch.randn(16, dtype=torch.float64, device=device, requires_grad=True)
        eps = 1e-6
        assert gradcheck(
            lambda x, w: MageRMSNorm.apply(x, w, eps),
            (x, weight),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestRMSNormDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        x = torch.randn(32, 64, device=device, dtype=dtype)
        weight = torch.randn(64, device=device, dtype=dtype)
        result = rmsnorm(x, weight)
        assert result.dtype == dtype

    def test_fp16_accuracy(self, device):
        x = torch.randn(32, 64, device=device, dtype=torch.float16)
        weight = torch.randn(64, device=device, dtype=torch.float16)
        result = rmsnorm(x, weight)
        expected = torch_rmsnorm(x, weight)
        # Looser tolerance for fp16
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
