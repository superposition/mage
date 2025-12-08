"""Profiler backends for nsys, ncu, and Triton native."""

from mage.profiler.backends.base import ProfilerBackend
from mage.profiler.backends.nsys import NsysBackend
from mage.profiler.backends.ncu import NcuBackend
from mage.profiler.backends.triton_profiler import TritonBackend, profile_triton

__all__ = ["ProfilerBackend", "NsysBackend", "NcuBackend", "TritonBackend", "profile_triton"]


def get_backend(name: str = "triton") -> ProfilerBackend:
    """Get a profiler backend by name.

    Args:
        name: Backend name ('triton', 'nsys', or 'ncu')
              Default is 'triton' which requires no special permissions.

    Returns:
        ProfilerBackend instance

    Raises:
        ValueError: If backend name is unknown
    """
    backends = {
        "triton": TritonBackend,
        "nsys": NsysBackend,
        "ncu": NcuBackend,
    }
    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(backends.keys())}")
    return backends[name]()
