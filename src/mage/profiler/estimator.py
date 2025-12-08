"""Metric estimation without hardware counters.

When ncu/nsys aren't available, we can still estimate many metrics
using indirect measurements and Triton kernel introspection.
"""

from __future__ import annotations

import torch
import triton

from dataclasses import dataclass
from typing import Any


@dataclass
class EstimatedMetrics:
    """Metrics that can be estimated without hardware counters."""

    # From timing
    duration_us: float
    throughput_gbps: float | None = None  # Estimated from data size / time

    # From kernel configuration (Triton provides these)
    num_warps: int | None = None
    num_stages: int | None = None
    shared_mem_bytes: int | None = None

    # From GPU queries
    registers_per_thread: int | None = None  # From cubin introspection

    # Calculated
    theoretical_occupancy: float | None = None  # Based on registers/shared
    arithmetic_intensity: float | None = None  # FLOPS/byte estimate
    roofline_bound: str | None = None  # "memory" or "compute"


def estimate_bandwidth(
    data_size_bytes: int,
    duration_us: float,
) -> float:
    """Estimate achieved bandwidth from data size and duration.

    Args:
        data_size_bytes: Total bytes read + written
        duration_us: Kernel duration in microseconds

    Returns:
        Estimated bandwidth in GB/s
    """
    if duration_us <= 0:
        return 0.0
    duration_s = duration_us / 1e6
    return (data_size_bytes / duration_s) / 1e9


def estimate_occupancy(
    num_warps: int,
    registers_per_thread: int,
    shared_mem_bytes: int,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    """Estimate theoretical occupancy from resource usage.

    This uses the occupancy calculator logic from CUDA.

    Args:
        num_warps: Warps per block
        registers_per_thread: Registers used per thread
        shared_mem_bytes: Shared memory per block
        gpu_name: GPU name for looking up limits

    Returns:
        Dict with occupancy estimates and limiting factors
    """
    # Default to Ada Lovelace (RTX 4090) specs
    max_warps_per_sm = 48
    max_blocks_per_sm = 24
    max_registers_per_sm = 65536
    max_shared_per_sm = 100 * 1024  # 100KB for RTX 4090
    warp_size = 32

    threads_per_block = num_warps * warp_size

    # Calculate limits
    # Register limit
    regs_per_block = registers_per_thread * threads_per_block
    if regs_per_block > 0:
        blocks_by_regs = max_registers_per_sm // regs_per_block
    else:
        blocks_by_regs = max_blocks_per_sm

    # Shared memory limit
    if shared_mem_bytes > 0:
        blocks_by_shared = max_shared_per_sm // shared_mem_bytes
    else:
        blocks_by_shared = max_blocks_per_sm

    # Warp limit
    blocks_by_warps = max_warps_per_sm // num_warps

    # Find the actual limit
    max_blocks = min(blocks_by_regs, blocks_by_shared, blocks_by_warps, max_blocks_per_sm)
    active_warps = max_blocks * num_warps
    occupancy = active_warps / max_warps_per_sm

    # Determine limiting factor
    if max_blocks == blocks_by_regs:
        limiting_factor = "registers"
    elif max_blocks == blocks_by_shared:
        limiting_factor = "shared_memory"
    elif max_blocks == blocks_by_warps:
        limiting_factor = "warps"
    else:
        limiting_factor = "blocks_per_sm"

    return {
        "occupancy": occupancy,
        "active_warps": active_warps,
        "max_blocks": max_blocks,
        "limiting_factor": limiting_factor,
        "blocks_by_regs": blocks_by_regs,
        "blocks_by_shared": blocks_by_shared,
        "blocks_by_warps": blocks_by_warps,
    }


def estimate_roofline_bound(
    duration_us: float,
    data_bytes: int,
    flops: int,
    peak_bandwidth_gbps: float = 1008.0,  # RTX 4090
    peak_tflops: float = 82.6,  # RTX 4090 FP32
) -> dict[str, Any]:
    """Estimate if kernel is memory or compute bound.

    Args:
        duration_us: Kernel duration
        data_bytes: Total bytes transferred
        flops: Estimated FLOP count
        peak_bandwidth_gbps: GPU peak bandwidth
        peak_tflops: GPU peak TFLOPS

    Returns:
        Dict with roofline analysis
    """
    duration_s = duration_us / 1e6

    # Achieved metrics
    achieved_bandwidth = (data_bytes / duration_s) / 1e9 if duration_s > 0 else 0
    achieved_tflops = (flops / duration_s) / 1e12 if duration_s > 0 else 0

    # Arithmetic intensity
    ai = flops / data_bytes if data_bytes > 0 else float('inf')

    # Ridge point
    ridge_point = (peak_tflops * 1e12) / (peak_bandwidth_gbps * 1e9)

    # Determine bound
    if ai < ridge_point:
        bound = "memory"
        # Memory-bound: performance limited by bandwidth
        theoretical_tflops = ai * peak_bandwidth_gbps / 1e3
        efficiency = achieved_bandwidth / peak_bandwidth_gbps
    else:
        bound = "compute"
        # Compute-bound: performance limited by FLOPS
        theoretical_tflops = peak_tflops
        efficiency = achieved_tflops / peak_tflops

    return {
        "bound": bound,
        "arithmetic_intensity": ai,
        "ridge_point": ridge_point,
        "achieved_bandwidth_gbps": achieved_bandwidth,
        "achieved_tflops": achieved_tflops,
        "efficiency": efficiency,
        "theoretical_peak_tflops": theoretical_tflops,
    }


def get_kernel_info(kernel_fn) -> dict[str, Any]:
    """Extract configuration from a Triton JIT function.

    This introspects the compiled kernel to get resource usage.
    """
    info = {
        "name": getattr(kernel_fn, "__name__", str(kernel_fn)),
    }

    # Try to get cached kernel info
    if hasattr(kernel_fn, "cache"):
        for key, compiled in kernel_fn.cache.items():
            if hasattr(compiled, "n_regs"):
                info["registers_per_thread"] = compiled.n_regs
            if hasattr(compiled, "shared"):
                info["shared_mem_bytes"] = compiled.shared
            if hasattr(compiled, "num_warps"):
                info["num_warps"] = compiled.num_warps
            break  # Just get first cached version

    return info


def profile_kernel_resources(
    kernel_fn,
    *args,
    grid: tuple,
    num_warps: int = 4,
    num_stages: int = 2,
    **kwargs,
) -> EstimatedMetrics:
    """Profile a kernel's resource usage and estimate metrics.

    This runs the kernel once to compile it and extracts info from the binary.
    """
    # Run kernel to ensure it's compiled
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    kernel_fn[grid](*args, num_warps=num_warps, num_stages=num_stages, **kwargs)
    torch.cuda.synchronize()

    # Timed run
    start.record()
    kernel_fn[grid](*args, num_warps=num_warps, num_stages=num_stages, **kwargs)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end)
    duration_us = duration_ms * 1000

    # Get kernel info
    kernel_info = get_kernel_info(kernel_fn)

    # Estimate occupancy if we have register info
    regs = kernel_info.get("registers_per_thread", 32)  # Default estimate
    shared = kernel_info.get("shared_mem_bytes", 0)

    occ_info = estimate_occupancy(num_warps, regs, shared)

    return EstimatedMetrics(
        duration_us=duration_us,
        num_warps=num_warps,
        num_stages=num_stages,
        shared_mem_bytes=shared,
        registers_per_thread=regs,
        theoretical_occupancy=occ_info["occupancy"],
    )


# GPU specifications for different architectures
GPU_SPECS = {
    # Ada Lovelace
    "RTX 4090": {
        "max_warps_per_sm": 48,
        "max_blocks_per_sm": 24,
        "max_registers_per_sm": 65536,
        "max_shared_per_sm": 100 * 1024,
        "sm_count": 128,
        "peak_bandwidth_gbps": 1008,
        "peak_fp32_tflops": 82.6,
        "peak_fp16_tflops": 165.2,
        "l2_cache_mb": 72,
    },
    "RTX 4080": {
        "max_warps_per_sm": 48,
        "max_blocks_per_sm": 24,
        "max_registers_per_sm": 65536,
        "max_shared_per_sm": 100 * 1024,
        "sm_count": 76,
        "peak_bandwidth_gbps": 717,
        "peak_fp32_tflops": 48.7,
        "peak_fp16_tflops": 97.5,
        "l2_cache_mb": 64,
    },
    # Ampere
    "A100": {
        "max_warps_per_sm": 64,
        "max_blocks_per_sm": 32,
        "max_registers_per_sm": 65536,
        "max_shared_per_sm": 164 * 1024,
        "sm_count": 108,
        "peak_bandwidth_gbps": 2039,  # HBM2e
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312,  # with sparsity
        "l2_cache_mb": 40,
    },
    # Hopper
    "H100": {
        "max_warps_per_sm": 64,
        "max_blocks_per_sm": 32,
        "max_registers_per_sm": 65536,
        "max_shared_per_sm": 228 * 1024,
        "sm_count": 132,
        "peak_bandwidth_gbps": 3350,  # HBM3
        "peak_fp32_tflops": 67,
        "peak_fp16_tflops": 1979,  # with sparsity
        "l2_cache_mb": 50,
    },
}


def detect_gpu_specs() -> dict[str, Any]:
    """Detect current GPU and return specs."""
    if not torch.cuda.is_available():
        return {}

    name = torch.cuda.get_device_name(0)

    # Try to match known GPUs
    for gpu_name, specs in GPU_SPECS.items():
        if gpu_name in name:
            return {"name": name, **specs}

    # Return basic info for unknown GPUs
    props = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "sm_count": props.multi_processor_count,
        "max_shared_per_sm": props.max_shared_memory_per_multiprocessor,
        "max_registers_per_sm": 65536,  # Assume
    }
