"""Cache hierarchy stress tests.

RTX 4090 specs:
- L1 cache: 128 KB per SM (128 SMs total)
- L2 cache: 72 MB total
- L1 bandwidth: ~200 bytes/clock/SM
- L2 bandwidth: ~6 TB/s aggregate
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


@triton.jit
def cache_probe_kernel(
    data_ptr,
    output_ptr,
    n_elements,
    stride,
    BLOCK_SIZE: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Probe cache at different strides to measure hit rates.

    Small stride + small data = L1 hits
    Medium stride + medium data = L2 hits
    Large stride + large data = DRAM
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Start with initial load
    idx = offs
    acc = tl.load(data_ptr + idx, mask=mask, other=0.0)

    # Repeated accesses at specified stride
    for _ in range(ITERATIONS):
        idx = (idx + stride) % n_elements
        data = tl.load(data_ptr + idx, mask=mask, other=0.0)
        acc = acc + data

    tl.store(output_ptr + offs, acc, mask=mask)


@triton.jit
def sequential_access_kernel(
    data_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential access pattern - best for cache."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    data = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(output_ptr + offs, data + 1.0, mask=mask)


@triton.jit
def random_access_kernel(
    data_ptr,
    index_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Random access pattern - worst for cache."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load random indices
    indices = tl.load(index_ptr + offs, mask=mask, other=0)
    # Random gather
    data = tl.load(data_ptr + indices, mask=mask, other=0.0)
    tl.store(output_ptr + offs, data, mask=mask)


@triton.jit
def cache_thrash_kernel(
    data_ptr,
    output_ptr,
    n_elements,
    working_set_size,
    BLOCK_SIZE: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Thrash cache by accessing more data than fits.

    working_set_size > L2 = DRAM bound
    working_set_size > L1 but < L2 = L2 bound
    working_set_size < L1 = L1 bound
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Access working set repeatedly
    for i in range(ITERATIONS):
        idx = (offs + i * BLOCK_SIZE) % working_set_size
        data = tl.load(data_ptr + idx, mask=idx < n_elements, other=0.0)
        acc = acc + data

    tl.store(output_ptr + offs, acc, mask=mask)


@dataclass
class CacheResult:
    """Result from cache benchmark."""
    name: str
    working_set_kb: float
    bandwidth_gbps: float
    duration_ms: float
    cache_level: str  # L1, L2, or DRAM
    hit_rate_estimate: float | None = None

    def __repr__(self) -> str:
        return (
            f"{self.name} ({self.working_set_kb:.0f} KB): "
            f"{self.bandwidth_gbps:.1f} GB/s [{self.cache_level}] - {self.duration_ms:.3f} ms"
        )


def benchmark_l1_cache(
    warmup: int = 5,
    iterations: int = 100,
    device: str = "cuda",
) -> list[CacheResult]:
    """Benchmark L1 cache performance at various working set sizes.

    L1 cache on RTX 4090: 128 KB per SM
    """
    # Working set sizes from 8KB to 256KB (spans L1 boundary)
    sizes_kb = [8, 16, 32, 64, 96, 128, 160, 192, 256]

    results = []
    BLOCK_SIZE = 256
    KERNEL_ITERATIONS = 100

    for size_kb in sizes_kb:
        n_elements = (size_kb * 1024) // 4  # float32
        working_set = torch.randn(n_elements, device=device, dtype=torch.float32)
        output = torch.empty_like(working_set)

        # Use few threads to keep working set per SM small
        grid = (1,)

        # Warmup
        for _ in range(warmup):
            cache_thrash_kernel[grid](
                working_set, output, n_elements, n_elements,
                BLOCK_SIZE=min(BLOCK_SIZE, n_elements),
                ITERATIONS=KERNEL_ITERATIONS,
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            cache_thrash_kernel[grid](
                working_set, output, n_elements, n_elements,
                BLOCK_SIZE=min(BLOCK_SIZE, n_elements),
                ITERATIONS=KERNEL_ITERATIONS,
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # Bytes accessed = working set * iterations
        bytes_accessed = n_elements * 4 * KERNEL_ITERATIONS
        bandwidth_gbps = (bytes_accessed / (duration_ms / 1000)) / 1e9

        # Determine cache level based on working set size
        if size_kb <= 128:
            cache_level = "L1"
        elif size_kb <= 72 * 1024:  # 72MB L2
            cache_level = "L2"
        else:
            cache_level = "DRAM"

        results.append(CacheResult(
            name=f"l1_probe_{size_kb}KB",
            working_set_kb=size_kb,
            bandwidth_gbps=bandwidth_gbps,
            duration_ms=duration_ms,
            cache_level=cache_level,
        ))

        del working_set, output
        torch.cuda.empty_cache()

    return results


def benchmark_l2_cache(
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
) -> list[CacheResult]:
    """Benchmark L2 cache performance at various working set sizes.

    L2 cache on RTX 4090: 72 MB
    """
    # Working set sizes from 1MB to 256MB (spans L2 boundary)
    sizes_mb = [1, 2, 4, 8, 16, 32, 48, 64, 72, 96, 128, 192, 256]

    results = []
    BLOCK_SIZE = 1024
    KERNEL_ITERATIONS = 10

    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // 4
        working_set = torch.randn(n_elements, device=device, dtype=torch.float32)
        output = torch.empty_like(working_set)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            cache_thrash_kernel[grid](
                working_set, output, n_elements, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ITERATIONS=KERNEL_ITERATIONS,
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            cache_thrash_kernel[grid](
                working_set, output, n_elements, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ITERATIONS=KERNEL_ITERATIONS,
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        bytes_accessed = n_elements * 4 * KERNEL_ITERATIONS
        bandwidth_gbps = (bytes_accessed / (duration_ms / 1000)) / 1e9

        if size_mb <= 72:
            cache_level = "L2"
        else:
            cache_level = "DRAM"

        results.append(CacheResult(
            name=f"l2_probe_{size_mb}MB",
            working_set_kb=size_mb * 1024,
            bandwidth_gbps=bandwidth_gbps,
            duration_ms=duration_ms,
            cache_level=cache_level,
        ))

        del working_set, output
        torch.cuda.empty_cache()

    return results


def benchmark_cache_thrash(
    size_mb: int = 256,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
) -> dict[str, CacheResult]:
    """Compare sequential vs random access patterns.

    Sequential: Good cache behavior
    Random: Cache thrashing
    """
    results = {}
    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 1024

    data = torch.randn(n_elements, device=device, dtype=torch.float32)
    output = torch.empty_like(data)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Sequential access
    for _ in range(warmup):
        sequential_access_kernel[grid](data, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        sequential_access_kernel[grid](data, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / iterations
    bytes_accessed = n_elements * 4 * 2  # read + write
    bandwidth_gbps = (bytes_accessed / (duration_ms / 1000)) / 1e9

    results["sequential"] = CacheResult(
        name="sequential",
        working_set_kb=size_mb * 1024,
        bandwidth_gbps=bandwidth_gbps,
        duration_ms=duration_ms,
        cache_level="streaming",
    )

    # Random access
    indices = torch.randint(0, n_elements, (n_elements,), device=device, dtype=torch.int64)

    for _ in range(warmup):
        random_access_kernel[grid](data, indices, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iterations):
        random_access_kernel[grid](data, indices, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / iterations
    # Random only does reads + writes output
    bytes_accessed = n_elements * 4 * 2
    bandwidth_gbps = (bytes_accessed / (duration_ms / 1000)) / 1e9

    results["random"] = CacheResult(
        name="random",
        working_set_kb=size_mb * 1024,
        bandwidth_gbps=bandwidth_gbps,
        duration_ms=duration_ms,
        cache_level="random",
    )

    del data, output, indices
    torch.cuda.empty_cache()

    return results
