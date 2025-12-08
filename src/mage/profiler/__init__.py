"""GPU profiler TUI for Triton/CUDA kernels."""

from mage.profiler.models import KernelMetric, ProfileSession
from mage.profiler.storage import ProfileDB
from mage.profiler.aggregator import MetricAggregator
from mage.profiler.columns import ColumnConfig, COLUMNS, DEFAULT_COLUMNS
from mage.profiler.tui import ProfilerTUI
from mage.profiler.backends import get_backend, NsysBackend, NcuBackend

__all__ = [
    "KernelMetric",
    "ProfileSession",
    "ProfileDB",
    "MetricAggregator",
    "ColumnConfig",
    "COLUMNS",
    "DEFAULT_COLUMNS",
    "ProfilerTUI",
    "get_backend",
    "NsysBackend",
    "NcuBackend",
]
