"""Fused MLP operations for efficient transformer FFN blocks.

Modern LLMs like LLaMA use a gated MLP structure:
    hidden = silu(gate_proj(x)) * up_proj(x)
    output = down_proj(hidden)

This module provides fused kernels to reduce memory bandwidth:
1. Fused SiLU-Multiply: silu(gate) * up in one kernel
2. Fused Gate-Up projection: compute both projections efficiently
"""

import torch
import triton
import triton.language as tl

from mage.matmul import matmul


@triton.jit
def _silu_mul_fwd_kernel(
    GATE,  # Gate tensor after projection
    UP,    # Up tensor after projection
    OUT,   # Output tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(GATE + offsets, mask=mask, other=0.0)
    up = tl.load(UP + offsets, mask=mask, other=0.0)

    # SiLU(gate) * up = gate * sigmoid(gate) * up
    # Use float32 for sigmoid accuracy
    gate_fp32 = gate.to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sigmoid_gate
    result = silu_gate * up.to(tl.float32)

    tl.store(OUT + offsets, result.to(gate.dtype), mask=mask)


@triton.jit
def _silu_mul_bwd_kernel(
    GRAD_OUT,  # Gradient from downstream
    GATE,      # Original gate values
    UP,        # Original up values
    GRAD_GATE, # Output: gradient w.r.t gate
    GRAD_UP,   # Output: gradient w.r.t up
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for fused SiLU-multiply.

    Forward: out = silu(gate) * up = gate * sigmoid(gate) * up

    d_out/d_gate = up * d(gate * sigmoid(gate))/d_gate
                 = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
                 = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))

    d_out/d_up = silu(gate) = gate * sigmoid(gate)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(GRAD_OUT + offsets, mask=mask, other=0.0)
    gate = tl.load(GATE + offsets, mask=mask, other=0.0)
    up = tl.load(UP + offsets, mask=mask, other=0.0)

    # Compute in float32 for accuracy
    grad_out_fp32 = grad_out.to(tl.float32)
    gate_fp32 = gate.to(tl.float32)
    up_fp32 = up.to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate_fp32)

    # d_out/d_up = silu(gate)
    silu_gate = gate_fp32 * sigmoid_gate
    grad_up = grad_out_fp32 * silu_gate

    # d_out/d_gate = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    grad_gate = grad_out_fp32 * up_fp32 * sigmoid_gate * (1.0 + gate_fp32 * (1.0 - sigmoid_gate))

    tl.store(GRAD_GATE + offsets, grad_gate.to(gate.dtype), mask=mask)
    tl.store(GRAD_UP + offsets, grad_up.to(up.dtype), mask=mask)


def silu_mul_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute silu(gate) * up efficiently.

    Args:
        gate: Gate projection output
        up: Up projection output

    Returns:
        silu(gate) * up
    """
    assert gate.shape == up.shape, f"Shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.is_contiguous() and up.is_contiguous()

    out = torch.empty_like(gate)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _silu_mul_fwd_kernel[grid](
        gate, up, out,
        n_elements,
        BLOCK_SIZE,
    )

    return out


def silu_mul_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for silu(gate) * up.

    Args:
        grad_out: Gradient from downstream
        gate: Original gate values
        up: Original up values

    Returns:
        (grad_gate, grad_up)
    """
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _silu_mul_bwd_kernel[grid](
        grad_out.contiguous(), gate, up,
        grad_gate, grad_up,
        n_elements,
        BLOCK_SIZE,
    )

    return grad_gate, grad_up


class SiLUMulFunc(torch.autograd.Function):
    """Autograd function for fused SiLU-multiply."""

    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return silu_mul_forward(gate, up)

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = silu_mul_backward(grad_out, gate, up)
        return grad_gate, grad_up


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up with autograd support.

    Args:
        gate: Gate projection output
        up: Up projection output

    Returns:
        silu(gate) * up
    """
    return SiLUMulFunc.apply(gate, up)


class FusedMLP(torch.nn.Module):
    """Fused MLP layer for LLaMA-style transformers.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Uses fused kernels to reduce memory bandwidth.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias, **factory_kwargs)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias, **factory_kwargs)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused operations.

        Args:
            x: Input tensor of shape (*, hidden_size)

        Returns:
            Output tensor of shape (*, hidden_size)
        """
        # Compute gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Fused SiLU-multiply
        hidden = silu_mul(gate, up)

        # Down projection
        return self.down_proj(hidden)


def fused_mlp_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: torch.Tensor = None,
    up_bias: torch.Tensor = None,
    down_bias: torch.Tensor = None,
) -> torch.Tensor:
    """Functional interface for fused MLP.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_size) or (batch, hidden_size)
        gate_weight: Gate projection weight (intermediate_size, hidden_size)
        up_weight: Up projection weight (intermediate_size, hidden_size)
        down_weight: Down projection weight (hidden_size, intermediate_size)
        gate_bias: Optional gate bias
        up_bias: Optional up bias
        down_bias: Optional down bias

    Returns:
        Output tensor of same shape as input
    """
    # Handle different input shapes
    original_shape = x.shape
    if x.dim() == 3:
        batch, seq_len, hidden = x.shape
        x = x.view(batch * seq_len, hidden)

    # Gate and up projections using optimized matmul
    gate = matmul(x, gate_weight.t().contiguous())
    up = matmul(x, up_weight.t().contiguous())

    if gate_bias is not None:
        gate = gate + gate_bias
    if up_bias is not None:
        up = up + up_bias

    # Fused SiLU-multiply
    hidden = silu_mul(gate, up)

    # Down projection
    out = matmul(hidden, down_weight.t().contiguous())
    if down_bias is not None:
        out = out + down_bias

    # Restore original shape
    if len(original_shape) == 3:
        out = out.view(original_shape[0], original_shape[1], -1)

    return out
