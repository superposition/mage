"""Register Triton kernels as torch custom ops for torch.compile compatibility."""

import torch
from torch import Tensor

from mage import kernels
from mage.matmul import matmul as matmul_impl

# =============================================================================
# Add
# =============================================================================


@torch.library.custom_op("mage::add", mutates_args=())
def add(x: Tensor, y: Tensor) -> Tensor:
    return kernels.add(x, y)


@add.register_fake
def _(x: Tensor, y: Tensor) -> Tensor:
    return torch.empty_like(x)


# =============================================================================
# FMA
# =============================================================================


@torch.library.custom_op("mage::fma", mutates_args=())
def fma(a: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return kernels.fma(a, x, y)


@fma.register_fake
def _(a: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return torch.empty_like(x)


# =============================================================================
# ReLU
# =============================================================================


@torch.library.custom_op("mage::relu", mutates_args=())
def relu(x: Tensor) -> Tensor:
    return kernels.relu(x)


@relu.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)


# =============================================================================
# Softmax
# =============================================================================


@torch.library.custom_op("mage::softmax", mutates_args=())
def softmax(x: Tensor) -> Tensor:
    return kernels.softmax(x)


@softmax.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)


# =============================================================================
# Matmul
# =============================================================================


@torch.library.custom_op("mage::matmul", mutates_args=())
def matmul(a: Tensor, b: Tensor) -> Tensor:
    return matmul_impl(a, b)


@matmul.register_fake
def _(a: Tensor, b: Tensor) -> Tensor:
    M, K = a.shape
    K, N = b.shape
    return torch.empty((M, N), device=a.device, dtype=a.dtype)
