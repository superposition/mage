"""Rotary Position Embeddings (RoPE) kernel.

RoPE applies a rotation to query and key vectors based on their position,
enabling relative position encoding without explicit position embeddings.

Reference: RoFormer: Enhanced Transformer with Rotary Position Embedding
https://arxiv.org/abs/2104.09864
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    X,  # Input tensor (batch*heads, seq, dim)
    OUT,  # Output tensor
    COS,  # Precomputed cos values (seq, dim//2)
    SIN,  # Precomputed sin values (seq, dim//2)
    seq_len,
    head_dim,
    stride_xbh, stride_xs, stride_xd,
    stride_obh, stride_os, stride_od,
    stride_cs, stride_cd,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embeddings.

    For each pair of dimensions (2i, 2i+1):
    x'[2i]   = x[2i] * cos - x[2i+1] * sin
    x'[2i+1] = x[2i] * sin + x[2i+1] * cos
    """
    # Program IDs: (batch*head, seq_pos)
    pid_bh = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # Base offset for this (batch_head, seq_pos)
    x_base = pid_bh * stride_xbh + pid_seq * stride_xs
    out_base = pid_bh * stride_obh + pid_seq * stride_os
    cos_base = pid_seq * stride_cs
    sin_base = pid_seq * stride_cs

    # Process all dimension pairs
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim // 2

    # Load x[2i] and x[2i+1]
    x_even = tl.load(X + x_base + 2 * offs * stride_xd, mask=mask, other=0.0)
    x_odd = tl.load(X + x_base + (2 * offs + 1) * stride_xd, mask=mask, other=0.0)

    # Load cos and sin for this position
    cos_val = tl.load(COS + cos_base + offs * stride_cd, mask=mask, other=0.0)
    sin_val = tl.load(SIN + sin_base + offs * stride_cd, mask=mask, other=0.0)

    # Apply rotation
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd = x_even * sin_val + x_odd * cos_val

    # Store results
    tl.store(OUT + out_base + 2 * offs * stride_od, out_even, mask=mask)
    tl.store(OUT + out_base + (2 * offs + 1) * stride_od, out_odd, mask=mask)


def precompute_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
    """Precompute the frequency tensor for RoPE.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for the frequency computation (default 10000)
        device: Device to place tensors on

    Returns:
        cos, sin: Tensors of shape (max_seq_len, dim // 2)
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE"

    # Compute frequencies: theta_i = 1 / (theta^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Compute position indices
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (seq_len, dim//2)
    angles = torch.outer(positions, freqs)

    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def rope_forward(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim)
           or (batch * num_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim // 2)
        sin: Sine values of shape (seq_len, head_dim // 2)

    Returns:
        Output tensor with RoPE applied, same shape as input
    """
    assert x.is_contiguous(), "Input must be contiguous"

    # Handle different input shapes
    if x.dim() == 4:
        batch, num_heads, seq_len, head_dim = x.shape
        x_flat = x.view(batch * num_heads, seq_len, head_dim)
    else:
        x_flat = x
        batch_heads, seq_len, head_dim = x_flat.shape

    assert head_dim % 2 == 0, "Head dimension must be even"
    assert cos.shape[0] >= seq_len, f"cos length {cos.shape[0]} < seq_len {seq_len}"

    # Slice cos/sin to actual sequence length
    cos = cos[:seq_len].contiguous()
    sin = sin[:seq_len].contiguous()

    # Output tensor
    out_flat = torch.empty_like(x_flat)

    # Block size must be power of 2 and cover head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    # Grid: one program per (batch*head, seq_pos)
    grid = (x_flat.shape[0], seq_len)

    _rope_fwd_kernel[grid](
        x_flat, out_flat, cos, sin,
        seq_len, head_dim,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        cos.stride(0), cos.stride(1),
        BLOCK_SIZE,
    )

    # Reshape back if needed
    if x.dim() == 4:
        return out_flat.view(batch, num_heads, seq_len, head_dim)
    return out_flat


# =============================================================================
# Autograd Function
# =============================================================================


class RoPEFunc(torch.autograd.Function):
    """Autograd wrapper for RoPE.

    The backward pass is simple: apply RoPE with negated sin values
    (rotation in opposite direction).
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return rope_forward(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        # Backward is the inverse rotation: use -sin
        grad_x = rope_forward(grad_out.contiguous(), cos, -sin)
        return grad_x, None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings with autograd support.

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (max_seq_len, head_dim // 2)
        sin: Sine values of shape (max_seq_len, head_dim // 2)

    Returns:
        Output tensor with RoPE applied
    """
    return RoPEFunc.apply(x, cos, sin)
