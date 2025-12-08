"""Flash Attention kernel - memory-efficient attention mechanism.

Based on FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
https://arxiv.org/abs/2307.08691
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Flash Attention forward kernel.

    Each program computes a BLOCK_M x BLOCK_K tile of the output.
    Uses online softmax to avoid materializing the full attention matrix.
    """
    # Program indices
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers to Q, K, V for this batch and head
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
             offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
             offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    # Load Q block (stays in registers throughout)
    mask_m = offs_m < M
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize online softmax statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)  # Output accumulator

    # Iterate over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N

        # Load K block
        k_ptrs_curr = K + pid_b * stride_kb + pid_h * stride_kh + \
                      offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs_curr, mask=mask_n[:, None], other=0.0)

        # Compute QK^T (BLOCK_M, BLOCK_N)
        # q is (BLOCK_M, BLOCK_K), k is (BLOCK_N, BLOCK_K)
        # We need q @ k^T = (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        # In Triton, tl.dot(a, b) expects a @ b, so we need k transposed
        # k^T is (BLOCK_K, BLOCK_N)
        k_t = tl.trans(k)  # (BLOCK_K, BLOCK_N)
        qk = tl.dot(q.to(tl.float16), k_t.to(tl.float16)).to(tl.float32)
        qk = qk * softmax_scale

        # Mask invalid positions
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)

        # Correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        # Update running sum
        l_i = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

        # Scale previous accumulator
        acc = acc * alpha[:, None]

        # Load V block and accumulate
        v_ptrs_curr = V + pid_b * stride_vb + pid_h * stride_vh + \
                      offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs_curr, mask=mask_n[:, None], other=0.0)

        # Compute attention weights (BLOCK_M, BLOCK_N)
        p = tl.exp(qk - m_new[:, None])

        # Accumulate: acc += P @ V
        # Both operands must be same dtype for dot
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)

        # Update max
        m_i = m_new

    # Normalize by sum
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
) -> torch.Tensor:
    """Flash Attention: memory-efficient attention mechanism.

    Args:
        q: Query tensor of shape (B, H, M, K) or (B, M, H, K) with head_first=False
        k: Key tensor of shape (B, H, N, K)
        v: Value tensor of shape (B, H, N, K)
        softmax_scale: Scale factor for softmax (default: 1/sqrt(K))

    Returns:
        Output tensor of shape (B, H, M, K)

    Note:
        All tensors must be contiguous and on CUDA.
        Uses online softmax to achieve O(N) memory instead of O(N²).
    """
    assert q.is_contiguous(), "Q must be contiguous"
    assert k.is_contiguous(), "K must be contiguous"
    assert v.is_contiguous(), "V must be contiguous"
    assert q.device.type == "cuda", "Flash attention requires CUDA"

    B, H, M, K = q.shape
    _, _, N, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (K ** 0.5)

    # Output tensor
    out = torch.empty_like(q)

    # Block sizes (tuned for common head dimensions)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = min(64, K)

    # Grid: one program per (M_block, batch*head)
    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return out


# =============================================================================
# Autograd Function
# =============================================================================


class FlashAttentionFunc(torch.autograd.Function):
    """Autograd wrapper for Flash Attention."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

        out = flash_attention(q, k, v, softmax_scale)
        ctx.save_for_backward(q, k, v, out)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        # For now, use PyTorch's autograd for backward
        # Full Flash Attention backward is complex and requires careful implementation
        grad_out = grad_out.contiguous()

        # Recompute attention weights
        B, H, M, K = q.shape
        N = k.shape[2]

        # Standard attention backward (not memory efficient)
        # This is a fallback - proper flash attention backward would recompute tiles
        q_2d = q.view(B * H, M, K)
        k_2d = k.view(B * H, N, K)
        v_2d = v.view(B * H, N, K)
        grad_out_2d = grad_out.view(B * H, M, K)

        # Recompute attention
        scores = torch.bmm(q_2d, k_2d.transpose(-2, -1)) * softmax_scale
        attn = torch.softmax(scores, dim=-1)

        # Gradient w.r.t. V: grad_V = A^T @ grad_out
        grad_v = torch.bmm(attn.transpose(-2, -1), grad_out_2d)

        # Gradient w.r.t. attention weights
        grad_attn = torch.bmm(grad_out_2d, v_2d.transpose(-2, -1))

        # Gradient through softmax: grad_scores = attn * (grad_attn - sum(attn * grad_attn))
        grad_scores = attn * (grad_attn - (attn * grad_attn).sum(dim=-1, keepdim=True))
        grad_scores = grad_scores * softmax_scale

        # Gradient w.r.t. Q: grad_Q = grad_scores @ K
        grad_q = torch.bmm(grad_scores, k_2d)

        # Gradient w.r.t. K: grad_K = grad_scores^T @ Q
        grad_k = torch.bmm(grad_scores.transpose(-2, -1), q_2d)

        # Reshape back
        grad_q = grad_q.view(B, H, M, K)
        grad_k = grad_k.view(B, H, N, K)
        grad_v = grad_v.view(B, H, N, K)

        return grad_q, grad_k, grad_v, None
