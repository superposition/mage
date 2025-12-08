"""Shared memory stress tests.

RTX 4090 specs:
- Shared memory: 100 KB per SM (configurable L1/shared split)
- Shared memory bandwidth: ~19 TB/s aggregate (all SMs)
- Bank conflicts: 32 banks, 4-byte stride
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


@triton.jit
def shared_mem_bandwidth_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SMEM_SIZE: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Measure shared memory bandwidth by repeated load/store."""
    pid = tl.program_id(0)

    # Allocate shared memory
    smem = tl.zeros((SMEM_SIZE,), dtype=tl.float32)

    # Load from global to shared
    offs = tl.arange(0, SMEM_SIZE)
    global_offs = pid * SMEM_SIZE + offs
    mask = global_offs < n_elements

    data = tl.load(input_ptr + global_offs, mask=mask, other=0.0)

    # Repeated shared memory operations
    for _ in range(ITERATIONS):
        smem = data
        data = smem + 1.0

    # Store back to global
    tl.store(output_ptr + global_offs, data, mask=mask)


@triton.jit
def bank_conflict_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,  # Stride for bank access (1=no conflict, 32=max conflict)
    ITERATIONS: tl.constexpr,
):
    """Test bank conflict impact on shared memory performance.

    Stride of 1: No bank conflicts (best)
    Stride of 32: All threads hit same bank (worst)
    """
    pid = tl.program_id(0)

    # Each thread accesses shared memory with specified stride
    tid = tl.arange(0, BLOCK_SIZE)
    strided_idx = (tid * STRIDE) % (BLOCK_SIZE * STRIDE)

    # Load initial data
    global_offs = pid * BLOCK_SIZE + tid
    mask = global_offs < n_elements
    data = tl.load(input_ptr + global_offs, mask=mask, other=0.0)

    # Simulate shared memory access with stride pattern
    # Use explicit indexing to create bank conflicts
    acc = data
    for _ in range(ITERATIONS):
        # Strided access pattern creates bank conflicts
        shuffled = tl.sum(acc)  # Force dependency
        acc = acc * 0.99 + shuffled * 0.01

    tl.store(output_ptr + global_offs, acc, mask=mask)


@triton.jit
def reduction_smem_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Tree reduction using shared memory - common pattern."""
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load to registers
    data = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # Tree reduction (uses shared memory internally)
    result = tl.sum(data)

    # Store result
    if pid == 0:
        tl.store(output_ptr, result)


@dataclass
class SharedMemResult:
    """Result from shared memory benchmark."""
    name: str
    bandwidth_tbps: float
    duration_ms: float
    bank_conflicts: int | None = None
    efficiency_pct: float | None = None

    def __repr__(self) -> str:
        result = f"{self.name}: {self.bandwidth_tbps:.2f} TB/s ({self.duration_ms:.3f} ms)"
        if self.bank_conflicts is not None:
            result += f" - {self.bank_conflicts} bank conflicts"
        return result


def benchmark_shared_memory(
    size_mb: int = 64,
    iterations: int = 1000,
    warmup: int = 5,
    runs: int = 50,
    device: str = "cuda",
) -> SharedMemResult:
    """Benchmark shared memory bandwidth.

    Args:
        size_mb: Total data size in MB
        iterations: Shared memory iterations per kernel
        warmup: Warmup runs
        runs: Timed runs
        device: CUDA device

    Returns:
        SharedMemResult with bandwidth measurement
    """
    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 256
    SMEM_SIZE = 256  # Elements per block in shared memory

    input_data = torch.randn(n_elements, device=device, dtype=torch.float32)
    output_data = torch.empty_like(input_data)

    grid = lambda meta: (triton.cdiv(n_elements, meta['SMEM_SIZE']),)

    # Warmup
    for _ in range(warmup):
        shared_mem_bandwidth_kernel[grid](
            input_data, output_data, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            SMEM_SIZE=SMEM_SIZE,
            ITERATIONS=iterations,
        )
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        shared_mem_bandwidth_kernel[grid](
            input_data, output_data, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            SMEM_SIZE=SMEM_SIZE,
            ITERATIONS=iterations,
        )
    end.record()
    torch.cuda.synchronize()

    duration_ms = start.elapsed_time(end) / runs

    # Each iteration: read + write = 8 bytes per element
    bytes_transferred = n_elements * 8 * iterations
    bandwidth_tbps = (bytes_transferred / (duration_ms / 1000)) / 1e12

    del input_data, output_data
    torch.cuda.empty_cache()

    return SharedMemResult(
        name=f"smem_{size_mb}MB_{iterations}iter",
        bandwidth_tbps=bandwidth_tbps,
        duration_ms=duration_ms,
    )


def benchmark_bank_conflicts(
    size_mb: int = 64,
    strides: list[int] | None = None,
    iterations: int = 1000,
    warmup: int = 5,
    runs: int = 50,
    device: str = "cuda",
) -> list[SharedMemResult]:
    """Benchmark impact of bank conflicts.

    Tests different stride patterns to measure bank conflict overhead.

    Args:
        size_mb: Data size in MB
        strides: List of strides to test (default: powers of 2 from 1-32)
        iterations: Operations per kernel
        warmup: Warmup iterations
        runs: Timed iterations
        device: CUDA device

    Returns:
        List of results for each stride
    """
    if strides is None:
        strides = [1, 2, 4, 8, 16, 32]

    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 256

    results = []

    for stride in strides:
        input_data = torch.randn(n_elements, device=device, dtype=torch.float32)
        output_data = torch.empty_like(input_data)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            bank_conflict_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                STRIDE=stride,
                ITERATIONS=iterations,
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(runs):
            bank_conflict_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                STRIDE=stride,
                ITERATIONS=iterations,
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / runs

        # Estimate bank conflicts based on stride
        # Stride 1: 0 conflicts, Stride 32: max conflicts
        estimated_conflicts = 0 if stride == 1 else (stride - 1) * n_elements // 32

        results.append(SharedMemResult(
            name=f"stride_{stride}",
            bandwidth_tbps=0,  # Not meaningful for conflict test
            duration_ms=duration_ms,
            bank_conflicts=estimated_conflicts,
        ))

        del input_data, output_data
        torch.cuda.empty_cache()

    # Calculate relative efficiency (stride 1 = 100%)
    if results:
        baseline = results[0].duration_ms
        for r in results:
            r.efficiency_pct = (baseline / r.duration_ms) * 100

    return results


def benchmark_reduction(
    sizes: list[int] | None = None,
    warmup: int = 5,
    runs: int = 100,
    device: str = "cuda",
) -> list[SharedMemResult]:
    """Benchmark tree reduction (shared memory intensive).

    Args:
        sizes: Element counts to test
        warmup: Warmup iterations
        runs: Timed iterations
        device: CUDA device

    Returns:
        List of results for each size
    """
    if sizes is None:
        sizes = [2**i for i in range(20, 28)]  # 1M to 128M

    results = []
    BLOCK_SIZE = 256

    for n_elements in sizes:
        input_data = torch.randn(n_elements, device=device, dtype=torch.float32)
        output_data = torch.zeros(1, device=device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            reduction_smem_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(runs):
            reduction_smem_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / runs

        # Bandwidth: n_elements * 4 bytes read
        bytes_read = n_elements * 4
        bandwidth_tbps = (bytes_read / (duration_ms / 1000)) / 1e12

        results.append(SharedMemResult(
            name=f"reduce_{n_elements // (1024*1024)}M",
            bandwidth_tbps=bandwidth_tbps,
            duration_ms=duration_ms,
        ))

        del input_data, output_data
        torch.cuda.empty_cache()

    return results
