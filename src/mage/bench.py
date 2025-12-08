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


def print_results(results: list[BenchResult], title: str | None = None) -> None:
    """Pretty print benchmark results."""
    if title:
        print(f"\n{title}")
    print("-" * 80)
    for r in results:
        print(r)
    print("-" * 80)


def bench_matmul_sweep(
    sizes: list[int] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> list[BenchResult]:
    """Benchmark matmul across multiple sizes.

    Args:
        sizes: List of matrix sizes (square matrices). Defaults to common sizes.
        device: Device to run on
        dtype: Data type for matrices
    """
    from mage import matmul

    if sizes is None:
        sizes = [128, 256, 512, 1024, 2048, 4096]

    results = []
    for size in sizes:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)

        result = bench(
            f"matmul-{size}",
            lambda a=a, b=b: matmul(a, b),
            lambda a=a, b=b: torch.matmul(a, b),
            a,
            b,
        )
        results.append(result)

    return results


def bench_backward(
    name: str,
    triton_fn: Callable,
    torch_fn: Callable,
    *inputs,
    warmup: int = 25,
    rep: int = 100,
) -> BenchResult:
    """Benchmark forward + backward pass.

    Args:
        name: Name for the benchmark
        triton_fn: Triton kernel wrapper function
        torch_fn: Equivalent PyTorch function
        *inputs: Input tensors (will be cloned with requires_grad=True)
        warmup: Number of warmup iterations
        rep: Number of benchmark iterations
    """
    size = inputs[0].numel() if inputs else 0

    # Clone inputs for Triton test
    triton_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
    # Clone inputs for PyTorch test
    torch_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]

    def triton_fwd_bwd():
        for inp in triton_inputs:
            if inp.grad is not None:
                inp.grad.zero_()
        out = triton_fn(*triton_inputs)
        out.sum().backward()

    def torch_fwd_bwd():
        for inp in torch_inputs:
            if inp.grad is not None:
                inp.grad.zero_()
        out = torch_fn(*torch_inputs)
        out.sum().backward()

    triton_ms = triton.testing.do_bench(triton_fwd_bwd, warmup=warmup, rep=rep)
    torch_ms = triton.testing.do_bench(torch_fwd_bwd, warmup=warmup, rep=rep)

    return BenchResult(f"{name}_fwd+bwd", size, triton_ms, torch_ms)


def bench_elementwise_sweep(
    name: str,
    triton_fn: Callable,
    torch_fn: Callable,
    sizes: list[int] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_inputs: int = 2,
) -> list[BenchResult]:
    """Benchmark elementwise ops across multiple sizes.

    Args:
        name: Name for the benchmark
        triton_fn: Triton kernel wrapper function
        torch_fn: Equivalent PyTorch function
        sizes: List of tensor sizes
        device: Device to run on
        dtype: Data type for tensors
        num_inputs: Number of input tensors
    """
    if sizes is None:
        sizes = [2**i for i in range(12, 25)]  # 4K to 16M elements

    results = []
    for size in sizes:
        inputs = [torch.randn(size, device=device, dtype=dtype) for _ in range(num_inputs)]
        result = bench(name, lambda *args: triton_fn(*args), lambda *args: torch_fn(*args), *inputs)
        results.append(result)

    return results


def bench_all(device: str = "cuda") -> dict[str, list[BenchResult]]:
    """Run comprehensive benchmarks on all operations.

    Returns:
        Dictionary mapping operation names to benchmark results
    """
    from mage import add, fma, matmul, relu, softmax

    all_results = {}

    # Elementwise ops
    print("Benchmarking elementwise operations...")
    sizes = [2**i for i in range(14, 23)]  # 16K to 4M elements

    all_results["add"] = bench_elementwise_sweep(
        "add", add, lambda x, y: x + y, sizes=sizes, device=device
    )

    all_results["relu"] = []
    for size in sizes:
        x = torch.randn(size, device=device)
        all_results["relu"].append(bench("relu", relu, torch.relu, x))

    all_results["fma"] = []
    for size in sizes:
        a = torch.randn(size, device=device)
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        all_results["fma"].append(bench("fma", fma, lambda a, x, y: a * x + y, a, x, y))

    # Softmax
    print("Benchmarking softmax...")
    all_results["softmax"] = []
    for rows in [32, 64, 128, 256, 512, 1024]:
        cols = 1024
        x = torch.randn(rows, cols, device=device)
        all_results["softmax"].append(
            bench(f"softmax-{rows}x{cols}", softmax, lambda x: torch.softmax(x, dim=1), x)
        )

    # Matmul
    print("Benchmarking matmul...")
    all_results["matmul"] = bench_matmul_sweep(device=device)

    # Backward passes
    print("Benchmarking backward passes...")
    all_results["backward"] = []

    # Add backward
    x = torch.randn(1_000_000, device=device)
    y = torch.randn(1_000_000, device=device)
    all_results["backward"].append(bench_backward("add", add, lambda a, b: a + b, x, y))

    # ReLU backward
    x = torch.randn(1_000_000, device=device)
    all_results["backward"].append(bench_backward("relu", relu, torch.relu, x))

    # Matmul backward
    a = torch.randn(512, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 512, device=device, dtype=torch.float16)
    all_results["backward"].append(bench_backward("matmul", matmul, torch.matmul, a, b))

    return all_results
