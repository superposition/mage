"""Memory bandwidth stress tests.

RTX 4090 specs:
- DRAM bandwidth: 1008 GB/s
- L2 cache: 72 MB
- L1 cache: 128 KB per SM (128 SMs)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Callable


@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple memory copy - measures raw bandwidth."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def read_only_kernel(
    src_ptr,
    dst_ptr,  # Single element for reduction
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Read-heavy workload - measures read bandwidth."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Read and accumulate to avoid compiler optimization
    data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(data)

    # Single atomic to avoid write bandwidth
    if pid == 0:
        tl.atomic_add(dst_ptr, acc)


@triton.jit
def write_only_kernel(
    dst_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    """Write-heavy workload - measures write bandwidth."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def strided_read_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided access - tests coalescing efficiency."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Strided access pattern
    strided_offsets = offsets * stride
    mask = strided_offsets < n_elements

    data = tl.load(src_ptr + strided_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, data, mask=offsets < (n_elements // stride))


@triton.jit
def vectorized_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized load/store with float4 - maximum bandwidth."""
    pid = tl.program_id(0)
    # Process 4 elements at a time
    offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE) * 4

    for i in range(4):
        idx = offsets + i
        mask = idx < n_elements
        data = tl.load(src_ptr + idx, mask=mask)
        tl.store(dst_ptr + idx, data, mask=mask)


@dataclass
class BandwidthResult:
    """Result from bandwidth benchmark."""
    name: str
    size_bytes: int
    duration_ms: float
    bandwidth_gbps: float
    efficiency_pct: float  # vs theoretical peak

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.bandwidth_gbps:.1f} GB/s "
            f"({self.efficiency_pct:.1f}% of peak) - {self.duration_ms:.3f} ms"
        )


def benchmark_memory_bandwidth(
    sizes: list[int] | None = None,
    warmup: int = 5,
    iterations: int = 100,
    device: str = "cuda",
    peak_bandwidth_gbps: float = 1008.0,  # RTX 4090
) -> list[BandwidthResult]:
    """Benchmark raw memory copy bandwidth at various sizes.

    Args:
        sizes: List of sizes in bytes to test (default: 1MB to 4GB)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        device: CUDA device
        peak_bandwidth_gbps: Theoretical peak bandwidth

    Returns:
        List of BandwidthResult for each size
    """
    if sizes is None:
        # Default: 1MB to 4GB in powers of 2
        sizes = [2**i for i in range(20, 33)]  # 1MB to 4GB

    results = []
    BLOCK_SIZE = 1024

    for size_bytes in sizes:
        n_elements = size_bytes // 4  # float32

        src = torch.randn(n_elements, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # Bandwidth: read + write
        total_bytes = size_bytes * 2  # read src + write dst
        bandwidth_gbps = (total_bytes / (duration_ms / 1000)) / 1e9
        efficiency = (bandwidth_gbps / peak_bandwidth_gbps) * 100

        results.append(BandwidthResult(
            name=f"copy_{size_bytes // (1024*1024)}MB",
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            efficiency_pct=efficiency,
        ))

        del src, dst
        torch.cuda.empty_cache()

    return results


def benchmark_memory_coalescing(
    size_mb: int = 256,
    warmup: int = 5,
    iterations: int = 100,
    device: str = "cuda",
    peak_bandwidth_gbps: float = 1008.0,
) -> list[BandwidthResult]:
    """Compare coalesced vs non-coalesced memory access.

    Tests stride patterns from 1 (coalesced) to 32 (worst case).
    """
    results = []
    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 1024

    strides = [1, 2, 4, 8, 16, 32]

    for stride in strides:
        # Need larger source for strided access
        src_size = n_elements * stride
        src = torch.randn(src_size, device=device, dtype=torch.float32)
        dst = torch.empty(n_elements, device=device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            strided_read_kernel[grid](src, dst, src_size, stride, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            strided_read_kernel[grid](src, dst, src_size, stride, BLOCK_SIZE=BLOCK_SIZE)
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # Only count bytes actually read/written
        bytes_accessed = n_elements * 4 * 2  # read + write
        bandwidth_gbps = (bytes_accessed / (duration_ms / 1000)) / 1e9
        efficiency = (bandwidth_gbps / peak_bandwidth_gbps) * 100

        results.append(BandwidthResult(
            name=f"stride_{stride}",
            size_bytes=n_elements * 4,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            efficiency_pct=efficiency,
        ))

        del src, dst
        torch.cuda.empty_cache()

    return results


def benchmark_memory_strided(
    size_mb: int = 256,
    warmup: int = 5,
    iterations: int = 100,
    device: str = "cuda",
) -> dict[str, BandwidthResult]:
    """Benchmark different strided access patterns.

    Returns dict mapping pattern name to result.
    """
    results = {}
    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 1024

    # Test read-only vs write-only vs copy
    src = torch.randn(n_elements, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    acc = torch.zeros(1, device=device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Read-only benchmark
    for _ in range(warmup):
        read_only_kernel[grid](src, acc, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        read_only_kernel[grid](src, acc, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / iterations
    bytes_read = n_elements * 4
    bandwidth_gbps = (bytes_read / (duration_ms / 1000)) / 1e9

    results["read_only"] = BandwidthResult(
        name="read_only",
        size_bytes=bytes_read,
        duration_ms=duration_ms,
        bandwidth_gbps=bandwidth_gbps,
        efficiency_pct=(bandwidth_gbps / 1008.0) * 100,
    )

    # Write-only benchmark
    for _ in range(warmup):
        write_only_kernel[grid](dst, n_elements, 1.0, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iterations):
        write_only_kernel[grid](dst, n_elements, 1.0, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / iterations
    bytes_written = n_elements * 4
    bandwidth_gbps = (bytes_written / (duration_ms / 1000)) / 1e9

    results["write_only"] = BandwidthResult(
        name="write_only",
        size_bytes=bytes_written,
        duration_ms=duration_ms,
        bandwidth_gbps=bandwidth_gbps,
        efficiency_pct=(bandwidth_gbps / 1008.0) * 100,
    )

    # Copy (read + write) benchmark
    for _ in range(warmup):
        copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iterations):
        copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / iterations
    total_bytes = n_elements * 4 * 2
    bandwidth_gbps = (total_bytes / (duration_ms / 1000)) / 1e9

    results["copy"] = BandwidthResult(
        name="copy",
        size_bytes=total_bytes,
        duration_ms=duration_ms,
        bandwidth_gbps=bandwidth_gbps,
        efficiency_pct=(bandwidth_gbps / 1008.0) * 100,
    )

    del src, dst, acc
    torch.cuda.empty_cache()

    return results
