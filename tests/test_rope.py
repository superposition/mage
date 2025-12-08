import pytest
import torch

from mage import rope, precompute_freqs


def torch_rope(x, cos, sin):
    """Reference implementation of RoPE using PyTorch."""
    # x: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim // 2)
    batch, num_heads, seq_len, head_dim = x.shape

    # Slice cos/sin to actual sequence length
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # Split into even and odd dimensions
    x_even = x[..., 0::2]  # (batch, num_heads, seq_len, head_dim // 2)
    x_odd = x[..., 1::2]

    # Expand cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1)
    out = out.reshape(batch, num_heads, seq_len, head_dim)

    return out


class TestPrecomputeFreqs:
    def test_shape(self, device):
        dim = 64
        max_seq_len = 128
        cos, sin = precompute_freqs(dim, max_seq_len, device=device)
        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_cos_sin_range(self, device):
        cos, sin = precompute_freqs(64, 128, device=device)
        assert cos.min() >= -1.0 and cos.max() <= 1.0
        assert sin.min() >= -1.0 and sin.max() <= 1.0

    def test_first_position_is_zero_rotation(self, device):
        cos, sin = precompute_freqs(64, 128, device=device)
        # At position 0, angles should be 0, so cos=1, sin=0
        assert torch.allclose(cos[0], torch.ones_like(cos[0]), atol=1e-5)
        assert torch.allclose(sin[0], torch.zeros_like(sin[0]), atol=1e-5)

    def test_different_theta(self, device):
        cos1, sin1 = precompute_freqs(64, 128, theta=10000.0, device=device)
        cos2, sin2 = precompute_freqs(64, 128, theta=500000.0, device=device)
        # Different theta should produce different frequencies
        assert not torch.allclose(cos1, cos2)


class TestRoPEForward:
    def test_basic(self, device):
        batch, num_heads, seq_len, head_dim = 2, 4, 64, 32
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)

        result = rope(x, cos, sin)
        expected = torch_rope(x, cos, sin)

        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self, device):
        batch, num_heads, seq_len, head_dim = 2, 8, 128, 64
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        cos, sin = precompute_freqs(head_dim, 256, device=device)

        result = rope(x, cos, sin)
        assert result.shape == x.shape

    def test_different_seq_lengths(self, device):
        batch, num_heads, head_dim = 2, 4, 32
        cos, sin = precompute_freqs(head_dim, 512, device=device)

        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            result = rope(x, cos, sin)
            expected = torch_rope(x, cos, sin)
            assert torch.allclose(result, expected, atol=1e-4), f"Failed for seq_len={seq_len}"

    def test_position_zero_is_identity(self, device):
        # At position 0, RoPE should be identity (no rotation)
        batch, num_heads, head_dim = 2, 4, 32
        x = torch.randn(batch, num_heads, 1, head_dim, device=device)
        cos, sin = precompute_freqs(head_dim, 128, device=device)

        result = rope(x, cos, sin)
        # cos[0] = 1, sin[0] = 0, so result should equal input
        assert torch.allclose(result, x, atol=1e-5)

    def test_single_batch_single_head(self, device):
        x = torch.randn(1, 1, 64, 32, device=device)
        cos, sin = precompute_freqs(32, 128, device=device)

        result = rope(x, cos, sin)
        expected = torch_rope(x, cos, sin)
        assert torch.allclose(result, expected, atol=1e-4)


class TestRoPEBackward:
    def test_gradient_exists(self, device):
        batch, num_heads, seq_len, head_dim = 2, 4, 64, 32
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)

        result = rope(x, cos, sin)
        result.sum().backward()

        assert x.grad is not None

    def test_gradient_shape(self, device):
        batch, num_heads, seq_len, head_dim = 2, 4, 64, 32
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)

        result = rope(x, cos, sin)
        result.sum().backward()

        assert x.grad.shape == x.shape

    def test_matches_torch_backward(self, device):
        batch, num_heads, seq_len, head_dim = 2, 4, 32, 16
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)

        # Triton
        result = rope(x, cos, sin)
        result.sum().backward()
        triton_grad = x.grad.clone()

        # Reset gradient
        x.grad = None

        # PyTorch reference
        ref = torch_rope(x, cos, sin)
        ref.sum().backward()

        assert torch.allclose(triton_grad, x.grad, atol=1e-4, rtol=1e-4)


class TestRoPEDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        batch, num_heads, seq_len, head_dim = 2, 4, 64, 32
        x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)
        cos, sin = cos.to(dtype), sin.to(dtype)

        result = rope(x, cos, sin)
        assert result.dtype == dtype
