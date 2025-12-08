"""GPU stress test benchmarks for RTX 4090 and other high-end GPUs."""

from mage.stress.memory import (
    benchmark_memory_bandwidth,
    benchmark_memory_coalescing,
    benchmark_memory_strided,
)
from mage.stress.compute import (
    benchmark_compute_fp32,
    benchmark_compute_fp16,
    benchmark_compute_int8,
)
from mage.stress.cache import (
    benchmark_l1_cache,
    benchmark_l2_cache,
    benchmark_cache_thrash,
)
from mage.stress.shared import (
    benchmark_shared_memory,
    benchmark_bank_conflicts,
)
from mage.stress.roofline import (
    benchmark_roofline,
    plot_roofline,
)
from mage.stress.runner import run_stress_suite, StressResult

__all__ = [
    "benchmark_memory_bandwidth",
    "benchmark_memory_coalescing",
    "benchmark_memory_strided",
    "benchmark_compute_fp32",
    "benchmark_compute_fp16",
    "benchmark_compute_int8",
    "benchmark_l1_cache",
    "benchmark_l2_cache",
    "benchmark_cache_thrash",
    "benchmark_shared_memory",
    "benchmark_bank_conflicts",
    "benchmark_roofline",
    "plot_roofline",
    "run_stress_suite",
    "StressResult",
]
