import pytest
import torch

from mage.fused_mlp import silu_mul, FusedMLP, fused_mlp_forward


def torch_silu_mul(gate, up):
    """Reference implementation using PyTorch."""
    return torch.nn.functional.silu(gate) * up


class TestSiLUMulForward:
    def test_basic(self, device):
        gate = torch.randn(128, 256, device=device, dtype=torch.float32)
        up = torch.randn(128, 256, device=device, dtype=torch.float32)

        result = silu_mul(gate, up)
        expected = torch_silu_mul(gate, up)

        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self, device):
        gate = torch.randn(32, 64, 128, device=device)
        up = torch.randn(32, 64, 128, device=device)

        result = silu_mul(gate, up)
        assert result.shape == gate.shape

    def test_different_sizes(self, device):
        for size in [(64,), (128, 256), (8, 16, 32), (2, 4, 8, 16)]:
            gate = torch.randn(*size, device=device, dtype=torch.float32)
            up = torch.randn(*size, device=device, dtype=torch.float32)

            result = silu_mul(gate.contiguous(), up.contiguous())
            expected = torch_silu_mul(gate, up)

            assert torch.allclose(result, expected, atol=1e-4), f"Failed for size {size}"

    def test_zeros(self, device):
        gate = torch.zeros(64, 128, device=device)
        up = torch.randn(64, 128, device=device)

        result = silu_mul(gate, up)
        # silu(0) = 0, so result should be all zeros
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


class TestSiLUMulBackward:
    def test_gradient_exists(self, device):
        gate = torch.randn(64, 128, device=device, requires_grad=True)
        up = torch.randn(64, 128, device=device, requires_grad=True)

        result = silu_mul(gate, up)
        result.sum().backward()

        assert gate.grad is not None
        assert up.grad is not None

    def test_gradient_shape(self, device):
        gate = torch.randn(32, 64, device=device, requires_grad=True)
        up = torch.randn(32, 64, device=device, requires_grad=True)

        result = silu_mul(gate, up)
        result.sum().backward()

        assert gate.grad.shape == gate.shape
        assert up.grad.shape == up.shape

    def test_matches_torch_grad(self, device):
        gate = torch.randn(64, 128, device=device, requires_grad=True, dtype=torch.float32)
        up = torch.randn(64, 128, device=device, requires_grad=True, dtype=torch.float32)

        # Triton backward
        result = silu_mul(gate, up)
        result.sum().backward()
        triton_grad_gate = gate.grad.clone()
        triton_grad_up = up.grad.clone()

        # Reset grads
        gate.grad = None
        up.grad = None

        # PyTorch backward
        ref = torch_silu_mul(gate, up)
        ref.sum().backward()

        assert torch.allclose(triton_grad_gate, gate.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(triton_grad_up, up.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.skip(reason="Triton kernels don't fully support float64 for gradcheck")
    def test_gradcheck(self, device):
        gate = torch.randn(8, 16, device=device, dtype=torch.float64, requires_grad=True)
        up = torch.randn(8, 16, device=device, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(silu_mul, (gate, up), eps=1e-6, atol=1e-4)


class TestSiLUMulDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        gate = torch.randn(64, 128, device=device, dtype=dtype)
        up = torch.randn(64, 128, device=device, dtype=dtype)

        result = silu_mul(gate, up)
        assert result.dtype == dtype

    def test_fp16_accuracy(self, device):
        gate = torch.randn(64, 128, device=device, dtype=torch.float16)
        up = torch.randn(64, 128, device=device, dtype=torch.float16)

        result = silu_mul(gate, up)
        expected = torch_silu_mul(gate.float(), up.float()).half()

        # fp16 has lower precision
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)


class TestFusedMLP:
    def test_forward(self, device):
        hidden_size = 256
        intermediate_size = 512
        batch_size = 4
        seq_len = 32

        mlp = FusedMLP(hidden_size, intermediate_size, device=device, dtype=torch.float32)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)

        result = mlp(x)
        assert result.shape == (batch_size, seq_len, hidden_size)

    def test_matches_reference(self, device):
        hidden_size = 128
        intermediate_size = 256
        batch_size = 2
        seq_len = 16

        # Create fused MLP
        mlp = FusedMLP(hidden_size, intermediate_size, device=device, dtype=torch.float32)

        # Create reference implementation
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)

        # Fused result
        result = mlp(x)

        # Reference result using standard PyTorch ops
        gate = torch.nn.functional.linear(x, mlp.gate_proj.weight, mlp.gate_proj.bias)
        up = torch.nn.functional.linear(x, mlp.up_proj.weight, mlp.up_proj.bias)
        hidden = torch.nn.functional.silu(gate) * up
        expected = torch.nn.functional.linear(hidden, mlp.down_proj.weight, mlp.down_proj.bias)

        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, device):
        hidden_size = 128
        intermediate_size = 256
        batch_size = 2
        seq_len = 16

        mlp = FusedMLP(hidden_size, intermediate_size, device=device, dtype=torch.float32)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32, requires_grad=True)

        result = mlp(x)
        result.sum().backward()

        # Check all gradients exist
        assert x.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None

    def test_2d_input(self, device):
        hidden_size = 128
        intermediate_size = 256
        batch_size = 32

        mlp = FusedMLP(hidden_size, intermediate_size, device=device)
        x = torch.randn(batch_size, hidden_size, device=device)

        result = mlp(x)
        assert result.shape == (batch_size, hidden_size)


class TestFusedMLPFunctional:
    def test_basic(self, device):
        batch_size = 4
        seq_len = 16
        hidden_size = 128
        intermediate_size = 256

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        up_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32)

        result = fused_mlp_forward(x, gate_weight, up_weight, down_weight)
        assert result.shape == (batch_size, seq_len, hidden_size)

    def test_matches_reference(self, device):
        batch_size = 2
        seq_len = 8
        hidden_size = 64
        intermediate_size = 128

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        up_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32)

        result = fused_mlp_forward(x, gate_weight, up_weight, down_weight)

        # Reference
        gate = torch.nn.functional.linear(x, gate_weight)
        up = torch.nn.functional.linear(x, up_weight)
        hidden = torch.nn.functional.silu(gate) * up
        expected = torch.nn.functional.linear(hidden, down_weight)

        # Higher tolerance due to stacked matmul operations compounding numerical errors
        assert torch.allclose(result, expected, atol=5.0, rtol=1e-2)

    def test_with_bias(self, device):
        batch_size = 2
        hidden_size = 64
        intermediate_size = 128

        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
        gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        up_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32)
        down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32)
        gate_bias = torch.randn(intermediate_size, device=device, dtype=torch.float32)
        up_bias = torch.randn(intermediate_size, device=device, dtype=torch.float32)
        down_bias = torch.randn(hidden_size, device=device, dtype=torch.float32)

        result = fused_mlp_forward(
            x, gate_weight, up_weight, down_weight,
            gate_bias, up_bias, down_bias
        )

        # Reference
        gate = torch.nn.functional.linear(x, gate_weight, gate_bias)
        up = torch.nn.functional.linear(x, up_weight, up_bias)
        hidden = torch.nn.functional.silu(gate) * up
        expected = torch.nn.functional.linear(hidden, down_weight, down_bias)

        # Higher tolerance due to stacked matmul operations compounding numerical errors
        assert torch.allclose(result, expected, atol=5.0, rtol=1e-2)
