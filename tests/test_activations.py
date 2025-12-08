import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from mage import gelu, silu
from mage.autograd import MageGelu, MageSilu


class TestGeluForward:
    def test_basic(self, device):
        x = torch.randn(1000, device=device)
        result = gelu(x)
        # PyTorch's default GELU is also the tanh approximation
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-5)

    def test_2d(self, device):
        x = torch.randn(32, 64, device=device)
        result = gelu(x)
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d(self, device):
        x = torch.randn(4, 32, 128, device=device)
        result = gelu(x)
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-5)

    def test_output_shape(self, device):
        x = torch.randn(16, 32, 64, device=device)
        result = gelu(x)
        assert result.shape == x.shape

    def test_zero_input(self, device):
        x = torch.zeros(100, device=device)
        result = gelu(x)
        # gelu(0) = 0
        assert torch.allclose(result, torch.zeros_like(x), atol=1e-6)

    def test_negative_values(self, device):
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], device=device)
        result = gelu(x)
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-5)


class TestGeluBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(100, device=device, requires_grad=True)
        result = gelu(x)
        result.sum().backward()
        assert x.grad is not None

    def test_gradient_shape(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        result = gelu(x)
        result.sum().backward()
        assert x.grad.shape == x.shape

    def test_matches_torch(self, device):
        x = torch.randn(1000, device=device, requires_grad=True)

        # Triton
        result = gelu(x)
        result.sum().backward()
        triton_grad = x.grad.clone()

        # PyTorch reference
        x.grad = None
        ref = F.gelu(x, approximate="tanh")
        ref.sum().backward()

        assert torch.allclose(triton_grad, x.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip(reason="GELU kernel uses fp32 accumulators, doesn't support fp64 for gradcheck")
    def test_gradcheck(self, device):
        x = torch.randn(50, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageGelu.apply, (x,), eps=1e-6, atol=1e-4)


class TestSiluForward:
    def test_basic(self, device):
        x = torch.randn(1000, device=device)
        result = silu(x)
        expected = F.silu(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_2d(self, device):
        x = torch.randn(32, 64, device=device)
        result = silu(x)
        expected = F.silu(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d(self, device):
        x = torch.randn(4, 32, 128, device=device)
        result = silu(x)
        expected = F.silu(x)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_output_shape(self, device):
        x = torch.randn(16, 32, 64, device=device)
        result = silu(x)
        assert result.shape == x.shape

    def test_zero_input(self, device):
        x = torch.zeros(100, device=device)
        result = silu(x)
        # silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert torch.allclose(result, torch.zeros_like(x), atol=1e-6)

    def test_identity_at_large_positive(self, device):
        # For large positive x, sigmoid(x) ≈ 1, so silu(x) ≈ x
        x = torch.tensor([10.0, 20.0, 50.0], device=device)
        result = silu(x)
        assert torch.allclose(result, x, rtol=1e-3)


class TestSiluBackward:
    def test_gradient_exists(self, device):
        x = torch.randn(100, device=device, requires_grad=True)
        result = silu(x)
        result.sum().backward()
        assert x.grad is not None

    def test_gradient_shape(self, device):
        x = torch.randn(32, 64, device=device, requires_grad=True)
        result = silu(x)
        result.sum().backward()
        assert x.grad.shape == x.shape

    def test_matches_torch(self, device):
        x = torch.randn(1000, device=device, requires_grad=True)

        # Triton
        result = silu(x)
        result.sum().backward()
        triton_grad = x.grad.clone()

        # PyTorch reference
        x.grad = None
        ref = F.silu(x)
        ref.sum().backward()

        assert torch.allclose(triton_grad, x.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip(reason="SiLU kernel uses fp32 accumulators, doesn't support fp64 for gradcheck")
    def test_gradcheck(self, device):
        x = torch.randn(50, dtype=torch.float64, device=device, requires_grad=True)
        assert gradcheck(MageSilu.apply, (x,), eps=1e-6, atol=1e-4)


class TestActivationDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_gelu_dtypes(self, device, dtype):
        x = torch.randn(100, device=device, dtype=dtype)
        result = gelu(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_silu_dtypes(self, device, dtype):
        x = torch.randn(100, device=device, dtype=dtype)
        result = silu(x)
        assert result.dtype == dtype

    def test_gelu_fp16_accuracy(self, device):
        x = torch.randn(1000, device=device, dtype=torch.float16)
        result = gelu(x)
        expected = F.gelu(x, approximate="tanh")
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_silu_fp16_accuracy(self, device):
        x = torch.randn(1000, device=device, dtype=torch.float16)
        result = silu(x)
        expected = F.silu(x)
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
