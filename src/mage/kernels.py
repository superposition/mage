"""Triton kernels with autotuning for optimal performance."""

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Autotuned Vector Add
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],  # Re-tune when this changes
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


# -----------------------------------------------------------------------------
# Fused Multiply-Add: output = a * x + y (single kernel, no intermediate)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}) for bs in [128, 256, 512, 1024, 2048]
    ],
    key=["n_elements"],
)
@triton.jit
def fma_kernel(
    a_ptr, x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused multiply-add: output = a * x + y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a * x + y, mask=mask)


# -----------------------------------------------------------------------------
# Softmax (fused, numerically stable)
# -----------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Numerically stable softmax over rows."""
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # Store
    out_start = row_idx * output_row_stride
    tl.store(output_ptr + out_start + col_offsets, result, mask=mask)


# -----------------------------------------------------------------------------
# ReLU (simple activation)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048]],
    key=["n_elements"],
)
@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# -----------------------------------------------------------------------------
# Python wrappers (the public API)
# -----------------------------------------------------------------------------

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition on GPU."""
    output = torch.empty_like(x)
    n = output.numel()
    # Grid must use the actual BLOCK_SIZE from autotune
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n)
    return output


def fma(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused multiply-add: a * x + y (no intermediate allocation)."""
    output = torch.empty_like(x)
    n = output.numel()
    # Grid must use the actual BLOCK_SIZE from autotune
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fma_kernel[grid](a, x, y, output, n)
    return output


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax."""
    assert x.dim() == 2, "Expected 2D tensor"
    rows, cols = x.shape
    # Block size must be power of 2 >= cols
    BLOCK_SIZE = triton.next_power_of_2(cols)
    output = torch.empty_like(x)
    softmax_kernel[(rows,)](x, output, cols, x.stride(0), output.stride(0), BLOCK_SIZE)
    return output


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation."""
    output = torch.empty_like(x)
    n = output.numel()
    # Grid must use the actual BLOCK_SIZE from autotune
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n)
    return output
