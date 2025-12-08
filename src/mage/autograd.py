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


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048]],
    key=["n_elements"],
)
@triton.jit
def gelu_backward_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU backward kernel.

    gelu(x) = 0.5 * x * (1 + tanh(u)) where u = sqrt(2/pi) * (x + 0.044715 * x^3)
    dgelu/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx
    where du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Cast to float32 for computation accuracy
    grad_out_fp32 = grad_out.to(tl.float32)
    x_fp32 = x.to(tl.float32)

    # Forward computation
    x_cubed = x_fp32 * x_fp32 * x_fp32
    u = 0.7978845608028654 * (x_fp32 + 0.044715 * x_cubed)
    # tanh(u) = 2*sigmoid(2u) - 1
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0

    # Backward computation
    # du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
    du_dx = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x_fp32 * x_fp32)
    # sech²(u) = 1 - tanh²(u)
    sech2_u = 1.0 - tanh_u * tanh_u
    # dgelu/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx
    grad_x = grad_out_fp32 * (0.5 * (1.0 + tanh_u) + 0.5 * x_fp32 * sech2_u * du_dx)

    tl.store(grad_x_ptr + offsets, grad_x.to(x.dtype), mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048]],
    key=["n_elements"],
)
@triton.jit
def silu_backward_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU backward kernel.

    silu(x) = x * sigmoid(x)
    dsilu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Cast to float32 for computation accuracy
    grad_out_fp32 = grad_out.to(tl.float32)
    x_fp32 = x.to(tl.float32)

    sigmoid_x = tl.sigmoid(x_fp32)
    # dsilu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    grad_x = grad_out_fp32 * sigmoid_x * (1.0 + x_fp32 * (1.0 - sigmoid_x))

    tl.store(grad_x_ptr + offsets, grad_x.to(x.dtype), mask=mask)


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


class MageGelu(torch.autograd.Function):
    """Autograd wrapper for GELU activation."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return kernels.gelu(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        grad_x = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        gelu_backward_kernel[grid](grad_out, x, grad_x, n)
        return grad_x


class MageSilu(torch.autograd.Function):
    """Autograd wrapper for SiLU/Swish activation."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return kernels.silu(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        grad_x = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        silu_backward_kernel[grid](grad_out, x, grad_x, n)
        return grad_x


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
