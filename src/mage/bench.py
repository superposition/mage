"""Benchmarking utilities for comparing Triton kernels against PyTorch."""

from dataclasses import dataclass
from typing import Callable

import torch
import triton


@dataclass
class BenchResult:
    name: str
    size: int
    triton_ms: float
    torch_ms: float

    @property
    def speedup(self) -> float:
        return self.torch_ms / self.triton_ms

    def __str__(self) -> str:
        return (
            f"{self.name:12} | size={self.size:>10,} | "
            f"triton={self.triton_ms:>8.3f}ms | torch={self.torch_ms:>8.3f}ms | "
            f"speedup={self.speedup:>5.2f}x"
        )


def bench(
    name: str,
    triton_fn: Callable,
    torch_fn: Callable,
    *args,
    warmup: int = 25,
    rep: int = 100,
) -> BenchResult:
    """Benchmark a Triton kernel against PyTorch.

    Args:
        name: Name for the benchmark
        triton_fn: Triton kernel wrapper function
        torch_fn: Equivalent PyTorch function
        *args: Input tensors
        warmup: Number of warmup iterations
        rep: Number of benchmark iterations
    """
    size = args[0].numel() if args else 0

    triton_ms = triton.testing.do_bench(lambda: triton_fn(*args), warmup=warmup, rep=rep)
    torch_ms = triton.testing.do_bench(lambda: torch_fn(*args), warmup=warmup, rep=rep)

    return BenchResult(name, size, triton_ms, torch_ms)


def bench_sweep(
    name: str,
    triton_fn: Callable,
    torch_fn: Callable,
    sizes: list[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> list[BenchResult]:
    """Benchmark across multiple sizes."""
    results = []
    for size in sizes:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)
        results.append(bench(name, triton_fn, torch_fn, x, y))
    return results


def print_results(results: list[BenchResult]) -> None:
    """Pretty print benchmark results."""
    print("-" * 80)
    for r in results:
        print(r)
    print("-" * 80)
