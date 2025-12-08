"""Profiler backends for nsys and ncu."""

from mage.profiler.backends.base import ProfilerBackend
from mage.profiler.backends.nsys import NsysBackend
from mage.profiler.backends.ncu import NcuBackend

__all__ = ["ProfilerBackend", "NsysBackend", "NcuBackend"]


def get_backend(name: str = "nsys") -> ProfilerBackend:
    """Get a profiler backend by name.

    Args:
        name: Backend name ('nsys' or 'ncu')

    Returns:
        ProfilerBackend instance

    Raises:
        ValueError: If backend name is unknown
    """
    backends = {
        "nsys": NsysBackend,
        "ncu": NcuBackend,
    }
    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Available: {list(backends.keys())}")
    return backends[name]()
