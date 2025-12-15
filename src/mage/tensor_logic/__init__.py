"""Tensor Logic primitives built on top of Triton kernels.

The initial phase focuses on providing a join (generalized contraction)
primitive with a clean Python API that can later be backed by custom GPU
kernels.  For now we provide a validated wrapper around ``torch.einsum``
so we can iterate on the API, tests, and integration points before
writing the Triton implementation.
"""

from .join import (
	TensorJoinPlan,
	tensor_join,
	tensor_join_cache_clear,
	tensor_join_cache_info,
)

__all__ = [
	"TensorJoinPlan",
	"tensor_join",
	"tensor_join_cache_clear",
	"tensor_join_cache_info",
]
