"""Mage - Composable Triton CUDA kernels with autotuning."""

from mage import kernels
from mage.autograd import MageAdd, MageFma, MageMatmul, MageRelu, MageSoftmax
from mage.matmul import matmul as _matmul_forward


# Public API uses autograd versions for training compatibility
def add(x, y):
    """Element-wise addition with autograd support."""
    return MageAdd.apply(x, y)


def fma(a, x, y):
    """Fused multiply-add (a*x + y) with autograd support."""
    return MageFma.apply(a, x, y)


def relu(x):
    """ReLU activation with autograd support."""
    return MageRelu.apply(x)


# Softmax with autograd support
def softmax(x):
    """Row-wise softmax with autograd support."""
    return MageSoftmax.apply(x)


# Matmul with autograd support
def matmul(a, b, out_dtype=None):
    """Matrix multiplication with autograd support.

    Args:
        a: (M, K) tensor
        b: (K, N) tensor
        out_dtype: Optional output dtype. If specified and requires_grad is False,
                   uses the direct implementation. Otherwise uses autograd wrapper.
    """
    if out_dtype is not None and not (a.requires_grad or b.requires_grad):
        # Use direct implementation with out_dtype when no gradients needed
        return _matmul_forward(a, b, out_dtype=out_dtype)
    elif out_dtype is not None:
        # For gradient case, compute then cast
        result = MageMatmul.apply(a, b)
        return result.to(out_dtype)
    return MageMatmul.apply(a, b)


__all__ = ["add", "fma", "relu", "softmax", "matmul"]
