"""Autograd-compatible wrappers for Triton kernels."""

import torch
from torch import Tensor

import triton
import triton.language as tl

from mage import kernels


# =============================================================================
# Backward Kernels
# =============================================================================


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048]],
    key=["n_elements"],
)
@triton.jit
def relu_backward_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU backward: grad_x = grad_out * (x > 0)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Gradient is grad_out where x > 0, else 0
    grad_x = tl.where(x > 0, grad_out, 0.0)
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048]],
    key=["n_elements"],
)
@triton.jit
def fma_backward_kernel(
    grad_out_ptr,
    a_ptr,
    x_ptr,
    grad_a_ptr,
    grad_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """FMA backward: grad_a = grad_out * x, grad_x = grad_out * a"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    tl.store(grad_a_ptr + offsets, grad_out * x, mask=mask)
    tl.store(grad_x_ptr + offsets, grad_out * a, mask=mask)


# =============================================================================
# Autograd Functions
# =============================================================================


class MageAdd(torch.autograd.Function):
    """Autograd wrapper for element-wise addition."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        # No need to save tensors - gradients are trivial
        return kernels.add(x, y)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        # d(x+y)/dx = 1, d(x+y)/dy = 1
        return grad_out.clone(), grad_out.clone()


class MageRelu(torch.autograd.Function):
    """Autograd wrapper for ReLU with fused backward kernel."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return kernels.relu(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        # Ensure contiguous for Triton kernel
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        grad_x = torch.empty_like(x)
        n = x.numel()
        # Grid must use the actual BLOCK_SIZE from autotune
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        relu_backward_kernel[grid](grad_out, x, grad_x, n)
        return grad_x


class MageFma(torch.autograd.Function):
    """Autograd wrapper for fused multiply-add: a * x + y"""

    @staticmethod
    def forward(ctx, a: Tensor, x: Tensor, y: Tensor) -> Tensor:
        ctx.save_for_backward(a, x)
        return kernels.fma(a, x, y)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a, x = ctx.saved_tensors
        # Ensure contiguous for Triton kernel
        grad_out = grad_out.contiguous()
        a = a.contiguous()
        x = x.contiguous()
        n = a.numel()

        grad_a = torch.empty_like(a)
        grad_x = torch.empty_like(x)

        # Grid must use the actual BLOCK_SIZE from autotune
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        fma_backward_kernel[grid](grad_out, a, x, grad_a, grad_x, n)

        # grad_y = grad_out (identity)
        grad_y = grad_out.clone()

        return grad_a, grad_x, grad_y


# =============================================================================
# Softmax Backward
# =============================================================================


@triton.jit
def softmax_backward_kernel(
    grad_out_ptr,
    y_ptr,
    grad_x_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax backward: grad_x = y * (grad_out - sum(grad_out * y))

    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load grad_out and forward output y for this row
    grad_out = tl.load(grad_out_ptr + row_start + col_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Compute dot product: sum(grad_out * y)
    dot_product = tl.sum(grad_out * y, axis=0)

    # Gradient: y * (grad_out - dot_product)
    grad_x = y * (grad_out - dot_product)

    tl.store(grad_x_ptr + row_start + col_offsets, grad_x, mask=mask)


class MageSoftmax(torch.autograd.Function):
    """Autograd wrapper for row-wise softmax."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        y = kernels.softmax(x)
        ctx.save_for_backward(y)  # Save output, not input
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (y,) = ctx.saved_tensors
        assert y.dim() == 2, "Softmax backward requires 2D tensor"

        # Ensure contiguous for Triton kernel
        grad_out = grad_out.contiguous()
        y = y.contiguous()

        rows, cols = y.shape
        BLOCK_SIZE = triton.next_power_of_2(cols)

        grad_x = torch.empty_like(y)
        softmax_backward_kernel[(rows,)](
            grad_out,
            y,
            grad_x,
            cols,
            y.stride(0),
            BLOCK_SIZE,
        )
        return grad_x


# =============================================================================
# Matmul Autograd (reuses forward kernel)
# =============================================================================


class MageMatmul(torch.autograd.Function):
    """Autograd wrapper for matrix multiplication.

    Forward:  C = A @ B
    Backward: grad_A = grad_C @ B.T
              grad_B = A.T @ grad_C
    """

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        from mage.matmul import matmul as matmul_forward

        ctx.save_for_backward(a, b)
        return matmul_forward(a, b)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        from mage.matmul import matmul as matmul_forward

        a, b = ctx.saved_tensors

        # Ensure contiguous for our kernel
        grad_out = grad_out.contiguous()

        # grad_A = grad_C @ B.T
        b_t = b.t().contiguous()
        grad_a = matmul_forward(grad_out, b_t)

        # grad_B = A.T @ grad_C
        a_t = a.t().contiguous()
        grad_b = matmul_forward(a_t, grad_out)

        return grad_a, grad_b
