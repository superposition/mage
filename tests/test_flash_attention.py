import pytest
import torch
import torch.nn.functional as F

from mage import flash_attention


def torch_attention(q, k, v, softmax_scale=None):
    """Reference implementation using PyTorch."""
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn = F.softmax(scores, dim=-1)
    # (B, H, M, N) @ (B, H, N, K) -> (B, H, M, K)
    out = torch.matmul(attn, v)
    return out


class TestFlashAttentionForward:
    def test_basic(self, device):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float16)

        result = flash_attention(q, k, v)
        expected = torch_attention(q, k, v)

        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_output_shape(self, device):
        B, H, M, N, K = 2, 8, 128, 128, 64
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float16)

        result = flash_attention(q, k, v)
        assert result.shape == (B, H, M, K)

    def test_different_seq_lengths(self, device):
        B, H, K = 2, 4, 32
        for M, N in [(64, 64), (64, 128), (128, 64), (32, 256)]:
            q = torch.randn(B, H, M, K, device=device, dtype=torch.float16)
            k = torch.randn(B, H, N, K, device=device, dtype=torch.float16)
            v = torch.randn(B, H, N, K, device=device, dtype=torch.float16)

            result = flash_attention(q, k, v)
            expected = torch_attention(q, k, v)

            assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2), f"Failed for M={M}, N={N}"

    def test_custom_scale(self, device):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float16)

        scale = 0.1
        result = flash_attention(q, k, v, softmax_scale=scale)
        expected = torch_attention(q, k, v, softmax_scale=scale)

        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_single_batch_single_head(self, device):
        B, H, M, N, K = 1, 1, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float16)

        result = flash_attention(q, k, v)
        expected = torch_attention(q, k, v)

        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_fp32(self, device):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float32)

        result = flash_attention(q, k, v)
        expected = torch_attention(q, k, v)

        assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)


class TestFlashAttentionBackward:
    def test_gradient_exists(self, device):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)

        result = flash_attention(q, k, v)
        result.sum().backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_gradient_shapes(self, device):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)

        result = flash_attention(q, k, v)
        result.sum().backward()

        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

    def test_matches_torch_backward(self, device):
        B, H, M, N, K = 2, 4, 32, 32, 16
        q = torch.randn(B, H, M, K, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, N, K, device=device, dtype=torch.float32, requires_grad=True)

        # Flash attention
        result = flash_attention(q, k, v)
        result.sum().backward()
        flash_grad_q = q.grad.clone()
        flash_grad_k = k.grad.clone()
        flash_grad_v = v.grad.clone()

        # Reset gradients
        q.grad = k.grad = v.grad = None

        # PyTorch reference
        ref = torch_attention(q, k, v)
        ref.sum().backward()

        assert torch.allclose(flash_grad_q, q.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(flash_grad_k, k.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(flash_grad_v, v.grad, atol=1e-3, rtol=1e-3)


class TestFlashAttentionDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        B, H, M, N, K = 2, 4, 64, 64, 32
        q = torch.randn(B, H, M, K, device=device, dtype=dtype)
        k = torch.randn(B, H, N, K, device=device, dtype=dtype)
        v = torch.randn(B, H, N, K, device=device, dtype=dtype)

        result = flash_attention(q, k, v)
        assert result.dtype == dtype
