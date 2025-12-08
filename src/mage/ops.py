"""Register Triton kernels as torch custom ops for torch.compile compatibility."""

import torch
from torch import Tensor

from mage import kernels

# Register custom ops in the "mage" namespace
# This makes them work with torch.compile and torch.export

@torch.library.custom_op("mage::add", mutates_args=())
def add(x: Tensor, y: Tensor) -> Tensor:
    return kernels.add(x, y)


@add.register_fake
def _(x: Tensor, y: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mage::fma", mutates_args=())
def fma(a: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return kernels.fma(a, x, y)


@fma.register_fake
def _(a: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mage::softmax", mutates_args=())
def softmax(x: Tensor) -> Tensor:
    return kernels.softmax(x)


@softmax.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mage::relu", mutates_args=())
def relu(x: Tensor) -> Tensor:
    return kernels.relu(x)


@relu.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)
