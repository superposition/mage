"""Compute throughput stress tests.

RTX 4090 specs:
- FP32: 82.6 TFLOPS
- FP16: 165.2 TFLOPS (with tensor cores: 330.4 TFLOPS)
- INT8: 660.6 TOPS (tensor cores)
- TF32: 165.2 TFLOPS (tensor cores)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


@triton.jit
def fma_stress_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OPS_PER_THREAD: tl.constexpr,
):
    """FMA-heavy kernel - measures peak FLOPS.

    Each thread does OPS_PER_THREAD fused multiply-adds.
    2 FLOPS per FMA instruction.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Initialize accumulators with varied values to prevent optimization
    a = tl.load(output_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = a * 0.99 + 0.01
    c = a * 0.98 + 0.02
    d = a * 0.97 + 0.03

    # Chain of dependent FMAs to stress ALU
    for _ in range(OPS_PER_THREAD):
        a = a * b + c
        b = b * c + d
        c = c * d + a
        d = d * a + b

    # Store to prevent dead code elimination
    result = a + b + c + d
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def fma_stress_fp16_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OPS_PER_THREAD: tl.constexpr,
):
    """FP16 FMA-heavy kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # FP16 computation
    a = tl.load(output_ptr + offsets, mask=mask, other=1.0).to(tl.float16)
    b = (a * 0.99 + 0.01).to(tl.float16)
    c = (a * 0.98 + 0.02).to(tl.float16)
    d = (a * 0.97 + 0.03).to(tl.float16)

    for _ in range(OPS_PER_THREAD):
        a = a * b + c
        b = b * c + d
        c = c * d + a
        d = d * a + b

    result = (a + b + c + d).to(tl.float32)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def matmul_stress_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul for compute stress testing."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@dataclass
class ComputeResult:
    """Result from compute benchmark."""
    name: str
    dtype: str
    size: int
    duration_ms: float
    tflops: float
    efficiency_pct: float  # vs theoretical peak

    def __repr__(self) -> str:
        return (
            f"{self.name} ({self.dtype}): {self.tflops:.2f} TFLOPS "
            f"({self.efficiency_pct:.1f}% of peak) - {self.duration_ms:.3f} ms"
        )


def benchmark_compute_fp32(
    sizes: list[int] | None = None,
    ops_per_thread: int = 1000,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
    peak_tflops: float = 82.6,  # RTX 4090 FP32
) -> list[ComputeResult]:
    """Benchmark FP32 compute throughput.

    Args:
        sizes: Number of threads to launch
        ops_per_thread: FMA operations per thread
        warmup: Warmup iterations
        iterations: Timed iterations
        device: CUDA device
        peak_tflops: Theoretical peak TFLOPS

    Returns:
        List of ComputeResult for each size
    """
    if sizes is None:
        # Scale from 1M to 128M threads
        sizes = [2**i for i in range(20, 28)]

    results = []
    BLOCK_SIZE = 256

    for n_elements in sizes:
        output = torch.ones(n_elements, device=device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            fma_stress_kernel[grid](
                output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_THREAD=ops_per_thread
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            fma_stress_kernel[grid](
                output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_THREAD=ops_per_thread
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # Calculate FLOPS:
        # 4 accumulators, each does 2 FMAs per iteration = 8 FMAs
        # Each FMA = 2 FLOPS
        flops_per_thread = ops_per_thread * 8 * 2
        total_flops = n_elements * flops_per_thread
        tflops = (total_flops / (duration_ms / 1000)) / 1e12

        efficiency = (tflops / peak_tflops) * 100

        results.append(ComputeResult(
            name=f"fma_{n_elements // (1024*1024)}M",
            dtype="fp32",
            size=n_elements,
            duration_ms=duration_ms,
            tflops=tflops,
            efficiency_pct=efficiency,
        ))

        del output
        torch.cuda.empty_cache()

    return results


def benchmark_compute_fp16(
    sizes: list[int] | None = None,
    ops_per_thread: int = 1000,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
    peak_tflops: float = 165.2,  # RTX 4090 FP16
) -> list[ComputeResult]:
    """Benchmark FP16 compute throughput."""
    if sizes is None:
        sizes = [2**i for i in range(20, 28)]

    results = []
    BLOCK_SIZE = 256

    for n_elements in sizes:
        output = torch.ones(n_elements, device=device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            fma_stress_fp16_kernel[grid](
                output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_THREAD=ops_per_thread
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            fma_stress_fp16_kernel[grid](
                output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_THREAD=ops_per_thread
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations
        flops_per_thread = ops_per_thread * 8 * 2
        total_flops = n_elements * flops_per_thread
        tflops = (total_flops / (duration_ms / 1000)) / 1e12

        efficiency = (tflops / peak_tflops) * 100

        results.append(ComputeResult(
            name=f"fma_{n_elements // (1024*1024)}M",
            dtype="fp16",
            size=n_elements,
            duration_ms=duration_ms,
            tflops=tflops,
            efficiency_pct=efficiency,
        ))

        del output
        torch.cuda.empty_cache()

    return results


def benchmark_compute_int8(
    sizes: list[int] | None = None,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
) -> list[ComputeResult]:
    """Benchmark INT8 matmul using tensor cores.

    Uses torch's int8 matmul which leverages tensor cores.
    """
    if sizes is None:
        # Matrix sizes for tensor core matmul
        sizes = [512, 1024, 2048, 4096, 8192]

    results = []

    for size in sizes:
        # INT8 matmul
        a = torch.randint(-128, 127, (size, size), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (size, size), device=device, dtype=torch.int8)

        # Warmup
        for _ in range(warmup):
            c = torch._int_mm(a, b)
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            c = torch._int_mm(a, b)
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # 2 * M * N * K operations for matmul
        ops = 2 * size * size * size
        tops = (ops / (duration_ms / 1000)) / 1e12

        results.append(ComputeResult(
            name=f"matmul_{size}x{size}",
            dtype="int8",
            size=size,
            duration_ms=duration_ms,
            tflops=tops,  # TOPS for int8
            efficiency_pct=(tops / 660.6) * 100,  # RTX 4090 INT8 TOPS
        ))

        del a, b, c
        torch.cuda.empty_cache()

    return results


def benchmark_matmul_tflops(
    sizes: list[int] | None = None,
    dtype: torch.dtype = torch.float16,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
) -> list[ComputeResult]:
    """Benchmark matmul TFLOPS using PyTorch (cublas).

    This is often the most realistic compute benchmark.
    """
    if sizes is None:
        sizes = [512, 1024, 2048, 4096, 8192, 16384]

    results = []
    dtype_name = str(dtype).split('.')[-1]

    # Peak TFLOPS for different dtypes on RTX 4090
    peak_tflops = {
        torch.float32: 82.6,
        torch.float16: 165.2,
        torch.bfloat16: 165.2,
    }.get(dtype, 82.6)

    for size in sizes:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # 2 * M * N * K FLOPS for matmul
        flops = 2 * size * size * size
        tflops = (flops / (duration_ms / 1000)) / 1e12

        results.append(ComputeResult(
            name=f"matmul_{size}x{size}",
            dtype=dtype_name,
            size=size,
            duration_ms=duration_ms,
            tflops=tflops,
            efficiency_pct=(tflops / peak_tflops) * 100,
        ))

        del a, b, c
        torch.cuda.empty_cache()

    return results
