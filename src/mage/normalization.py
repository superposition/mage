"""RMSNorm and LayerNorm kernels with autotuning."""

import torch
import triton
import triton.language as tl


# =============================================================================
# RMSNorm Forward Kernel
# =============================================================================


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols,
    eps,
    x_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward: y = x * rsqrt(mean(x²) + eps) * weight

    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Compute RMS: sqrt(mean(x²) + eps)
    x_squared = x * x
    mean_x_squared = tl.sum(x_squared, axis=0) / n_cols
    rrms = tl.rsqrt(mean_x_squared + eps)

    # Load weight and compute output
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    output = x * rrms * weight

    # Store
    out_start = row_idx * output_row_stride
    tl.store(output_ptr + out_start + col_offsets, output, mask=mask)


# =============================================================================
# RMSNorm Backward Kernel
# =============================================================================


@triton.jit
def rmsnorm_backward_kernel(
    grad_out_ptr,
    x_ptr,
    weight_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols,
    eps,
    grad_out_row_stride,
    x_row_stride,
    grad_x_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm backward for grad_x.

    Each program handles one row.
    grad_x = weight * rrms * (grad_out - x * mean(grad_out * x * rrms²) * rrms²)
           = rrms * (weight * grad_out - x * (1/n) * sum(grad_out * weight * x) * rrms²)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row data
    x_row_start = row_idx * x_row_stride
    grad_out_row_start = row_idx * grad_out_row_stride

    x = tl.load(x_ptr + x_row_start + col_offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + grad_out_row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)

    # Recompute rrms
    x_squared = x * x
    mean_x_squared = tl.sum(x_squared, axis=0) / n_cols
    rrms = tl.rsqrt(mean_x_squared + eps)

    # Compute grad_x
    # grad_x = rrms * weight * grad_out - rrms³ * x * mean(grad_out * weight * x)
    grad_out_weighted = grad_out * weight
    dot_product = tl.sum(grad_out_weighted * x, axis=0) / n_cols
    grad_x = rrms * grad_out_weighted - rrms * rrms * rrms * x * dot_product

    # Store grad_x
    grad_x_row_start = row_idx * grad_x_row_stride
    tl.store(grad_x_ptr + grad_x_row_start + col_offsets, grad_x, mask=mask)


@triton.jit
def rmsnorm_backward_weight_kernel(
    grad_out_ptr,
    x_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols,
    eps,
    grad_out_row_stride,
    x_row_stride,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """Compute grad_weight by accumulating over rows.

    grad_weight[j] = sum_i(grad_out[i,j] * x[i,j] * rrms[i])
    """
    col_block_idx = tl.program_id(0)
    col_offsets = col_block_idx * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < n_cols

    # Accumulate grad_weight over all rows
    grad_weight_acc = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.float32)

    for row_start in range(0, n_rows, BLOCK_SIZE_ROW):
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE_ROW)
        row_mask = row_offsets < n_rows

        for r in range(BLOCK_SIZE_ROW):
            row_idx = row_start + r
            if row_idx < n_rows:
                # Load x row and compute rrms
                x_row_start = row_idx * x_row_stride
                x = tl.load(x_ptr + x_row_start + col_offsets, mask=col_mask, other=0.0)

                x_squared = x * x
                mean_x_squared = tl.sum(x_squared, axis=0) / n_cols
                rrms = tl.rsqrt(mean_x_squared + eps)

                # Load grad_out row
                grad_out_row_start = row_idx * grad_out_row_stride
                grad_out = tl.load(grad_out_ptr + grad_out_row_start + col_offsets, mask=col_mask, other=0.0)

                # Accumulate: grad_weight += grad_out * x * rrms
                grad_weight_acc += grad_out * x * rrms

    # Store accumulated grad_weight
    tl.store(grad_weight_ptr + col_offsets, grad_weight_acc, mask=col_mask)


# =============================================================================
# Python Wrappers
# =============================================================================


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight

    Args:
        x: Input tensor of shape (*, hidden_size)
        weight: Learnable weight of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    assert x.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert x.shape[-1] == weight.shape[0], f"Hidden size mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    # Reshape to 2D for kernel
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    rows, cols = x_2d.shape

    output = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(cols)

    rmsnorm_kernel[(rows,)](
        x_2d,
        weight,
        output,
        cols,
        eps,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE,
    )

    return output.view(original_shape)


# =============================================================================
# Autograd Function
# =============================================================================


class MageRMSNorm(torch.autograd.Function):
    """Autograd wrapper for RMSNorm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rmsnorm(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        # Ensure contiguous
        grad_out = grad_out.contiguous()
        x = x.contiguous()

        # Reshape to 2D
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        grad_out_2d = grad_out.view(-1, grad_out.shape[-1])
        rows, cols = x_2d.shape

        BLOCK_SIZE = triton.next_power_of_2(cols)

        # Compute grad_x
        grad_x = torch.empty_like(x_2d)
        rmsnorm_backward_kernel[(rows,)](
            grad_out_2d,
            x_2d,
            weight,
            grad_x,
            None,  # grad_weight computed separately
            rows,
            cols,
            eps,
            grad_out_2d.stride(0),
            x_2d.stride(0),
            grad_x.stride(0),
            BLOCK_SIZE,
        )

        # Compute grad_weight using PyTorch (simpler and correct)
        # grad_weight[j] = sum_i(grad_out[i,j] * x[i,j] * rrms[i])
        x_squared = x_2d * x_2d
        mean_x_squared = x_squared.mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(mean_x_squared + eps)
        grad_weight = (grad_out_2d * x_2d * rrms).sum(dim=0)

        return grad_x.view(original_shape), grad_weight, None
