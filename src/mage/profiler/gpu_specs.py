"""GPU specifications database for theoretical performance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
import subprocess
import re


@dataclass
class GPUSpec:
    """Specifications for a GPU model."""

    name: str
    compute_capability: tuple[int, int]

    # Memory
    memory_gb: float
    memory_bandwidth_gbps: float  # GB/s
    memory_bus_width: int  # bits

    # Compute
    sm_count: int
    cuda_cores: int
    tensor_cores: int | None
    fp32_tflops: float
    fp16_tflops: float | None

    # Clocks (boost)
    gpu_clock_mhz: int
    memory_clock_mhz: int

    # Cache
    l2_cache_mb: float
    shared_mem_per_sm_kb: int

    @property
    def theoretical_occupancy_limit(self) -> dict:
        """Get occupancy limiters based on compute capability."""
        # Limits vary by compute capability
        if self.compute_capability >= (8, 0):  # Ampere+
            return {
                "max_threads_per_sm": 2048,
                "max_blocks_per_sm": 32,
                "max_warps_per_sm": 64,
                "max_registers_per_sm": 65536,
                "max_shared_mem_per_sm_kb": self.shared_mem_per_sm_kb,
                "max_threads_per_block": 1024,
                "warp_size": 32,
            }
        elif self.compute_capability >= (7, 0):  # Volta/Turing
            return {
                "max_threads_per_sm": 2048,
                "max_blocks_per_sm": 32,
                "max_warps_per_sm": 64,
                "max_registers_per_sm": 65536,
                "max_shared_mem_per_sm_kb": self.shared_mem_per_sm_kb,
                "max_threads_per_block": 1024,
                "warp_size": 32,
            }
        else:  # Pascal and older
            return {
                "max_threads_per_sm": 2048,
                "max_blocks_per_sm": 32,
                "max_warps_per_sm": 64,
                "max_registers_per_sm": 65536,
                "max_shared_mem_per_sm_kb": self.shared_mem_per_sm_kb,
                "max_threads_per_block": 1024,
                "warp_size": 32,
            }

    def arithmetic_intensity_balance(self) -> float:
        """Compute/memory balance point (FLOPs/byte)."""
        # At this AI, kernel is equally compute and memory bound
        return self.fp32_tflops * 1000 / self.memory_bandwidth_gbps


# Known GPU specifications
GPU_DATABASE: dict[str, GPUSpec] = {
    # Ada Lovelace (RTX 40 series)
    "RTX 4090": GPUSpec(
        name="RTX 4090",
        compute_capability=(8, 9),
        memory_gb=24,
        memory_bandwidth_gbps=1008,
        memory_bus_width=384,
        sm_count=128,
        cuda_cores=16384,
        tensor_cores=512,
        fp32_tflops=82.6,
        fp16_tflops=165.2,
        gpu_clock_mhz=2520,
        memory_clock_mhz=1313,
        l2_cache_mb=72,
        shared_mem_per_sm_kb=100,
    ),
    "RTX 4080": GPUSpec(
        name="RTX 4080",
        compute_capability=(8, 9),
        memory_gb=16,
        memory_bandwidth_gbps=717,
        memory_bus_width=256,
        sm_count=76,
        cuda_cores=9728,
        tensor_cores=304,
        fp32_tflops=48.7,
        fp16_tflops=97.5,
        gpu_clock_mhz=2505,
        memory_clock_mhz=1400,
        l2_cache_mb=64,
        shared_mem_per_sm_kb=100,
    ),
    "RTX 4070 Ti": GPUSpec(
        name="RTX 4070 Ti",
        compute_capability=(8, 9),
        memory_gb=12,
        memory_bandwidth_gbps=504,
        memory_bus_width=192,
        sm_count=60,
        cuda_cores=7680,
        tensor_cores=240,
        fp32_tflops=40.1,
        fp16_tflops=80.2,
        gpu_clock_mhz=2610,
        memory_clock_mhz=1313,
        l2_cache_mb=48,
        shared_mem_per_sm_kb=100,
    ),

    # Ampere (RTX 30 series)
    "RTX 3090": GPUSpec(
        name="RTX 3090",
        compute_capability=(8, 6),
        memory_gb=24,
        memory_bandwidth_gbps=936,
        memory_bus_width=384,
        sm_count=82,
        cuda_cores=10496,
        tensor_cores=328,
        fp32_tflops=35.6,
        fp16_tflops=71.2,
        gpu_clock_mhz=1695,
        memory_clock_mhz=1219,
        l2_cache_mb=6,
        shared_mem_per_sm_kb=100,
    ),
    "RTX 3080": GPUSpec(
        name="RTX 3080",
        compute_capability=(8, 6),
        memory_gb=10,
        memory_bandwidth_gbps=760,
        memory_bus_width=320,
        sm_count=68,
        cuda_cores=8704,
        tensor_cores=272,
        fp32_tflops=29.8,
        fp16_tflops=59.6,
        gpu_clock_mhz=1710,
        memory_clock_mhz=1188,
        l2_cache_mb=5,
        shared_mem_per_sm_kb=100,
    ),

    # Data center
    "A100 80GB": GPUSpec(
        name="A100 80GB",
        compute_capability=(8, 0),
        memory_gb=80,
        memory_bandwidth_gbps=2039,
        memory_bus_width=5120,
        sm_count=108,
        cuda_cores=6912,
        tensor_cores=432,
        fp32_tflops=19.5,
        fp16_tflops=312.0,  # With tensor cores
        gpu_clock_mhz=1410,
        memory_clock_mhz=1593,
        l2_cache_mb=40,
        shared_mem_per_sm_kb=164,
    ),
    "A100 40GB": GPUSpec(
        name="A100 40GB",
        compute_capability=(8, 0),
        memory_gb=40,
        memory_bandwidth_gbps=1555,
        memory_bus_width=5120,
        sm_count=108,
        cuda_cores=6912,
        tensor_cores=432,
        fp32_tflops=19.5,
        fp16_tflops=312.0,
        gpu_clock_mhz=1410,
        memory_clock_mhz=1215,
        l2_cache_mb=40,
        shared_mem_per_sm_kb=164,
    ),
    "H100 SXM": GPUSpec(
        name="H100 SXM",
        compute_capability=(9, 0),
        memory_gb=80,
        memory_bandwidth_gbps=3350,
        memory_bus_width=5120,
        sm_count=132,
        cuda_cores=16896,
        tensor_cores=528,
        fp32_tflops=67.0,
        fp16_tflops=1979.0,  # With tensor cores
        gpu_clock_mhz=1830,
        memory_clock_mhz=2619,
        l2_cache_mb=50,
        shared_mem_per_sm_kb=228,
    ),

    # Turing (RTX 20 series)
    "RTX 2080 Ti": GPUSpec(
        name="RTX 2080 Ti",
        compute_capability=(7, 5),
        memory_gb=11,
        memory_bandwidth_gbps=616,
        memory_bus_width=352,
        sm_count=68,
        cuda_cores=4352,
        tensor_cores=544,
        fp32_tflops=13.4,
        fp16_tflops=26.9,
        gpu_clock_mhz=1545,
        memory_clock_mhz=875,
        l2_cache_mb=5.5,
        shared_mem_per_sm_kb=64,
    ),
}


def detect_gpu() -> GPUSpec | None:
    """Detect the current GPU and return its specs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        gpu_name = result.stdout.strip()

        # Try exact match first
        for key, spec in GPU_DATABASE.items():
            if key in gpu_name:
                return spec

        # Try fuzzy match
        gpu_name_lower = gpu_name.lower()
        for key, spec in GPU_DATABASE.items():
            if key.lower() in gpu_name_lower:
                return spec

        # Return None if not found - caller can create custom spec
        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_gpu_spec(name: str | None = None) -> GPUSpec | None:
    """Get GPU spec by name or detect current GPU."""
    if name:
        # Exact match
        if name in GPU_DATABASE:
            return GPU_DATABASE[name]
        # Fuzzy match
        name_lower = name.lower()
        for key, spec in GPU_DATABASE.items():
            if key.lower() in name_lower or name_lower in key.lower():
                return spec
        return None

    return detect_gpu()
