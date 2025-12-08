"""Stress test runner - runs all benchmarks and generates reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


@dataclass
class StressResult:
    """Complete results from stress test suite."""
    gpu_name: str
    timestamp: datetime
    memory_results: dict[str, Any] = field(default_factory=dict)
    compute_results: dict[str, Any] = field(default_factory=dict)
    cache_results: dict[str, Any] = field(default_factory=dict)
    shared_results: dict[str, Any] = field(default_factory=dict)
    roofline_results: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


def run_stress_suite(
    device: str = "cuda",
    quick: bool = False,
    verbose: bool = True,
) -> StressResult:
    """Run the complete stress test suite.

    Args:
        device: CUDA device
        quick: Run quick version (fewer iterations)
        verbose: Print progress

    Returns:
        StressResult with all benchmark data
    """
    console = Console()

    result = StressResult(
        gpu_name=torch.cuda.get_device_name(0),
        timestamp=datetime.now(),
    )

    start_time = time.time()

    # Import here to avoid circular imports
    from mage.stress.memory import (
        benchmark_memory_bandwidth,
        benchmark_memory_coalescing,
        benchmark_memory_strided,
    )
    from mage.stress.compute import (
        benchmark_compute_fp32,
        benchmark_compute_fp16,
        benchmark_matmul_tflops,
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

    iterations = 20 if quick else 100
    warmup = 2 if quick else 5

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        # Memory benchmarks
        task = progress.add_task("Memory bandwidth...", total=5)

        # Quick sizes for fast mode
        if quick:
            bw_sizes = [2**i for i in range(24, 29, 2)]  # 16MB to 256MB
        else:
            bw_sizes = [2**i for i in range(22, 31)]  # 4MB to 1GB

        result.memory_results["bandwidth"] = benchmark_memory_bandwidth(
            sizes=bw_sizes,
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.memory_results["coalescing"] = benchmark_memory_coalescing(
            size_mb=128 if quick else 256,
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.memory_results["strided"] = benchmark_memory_strided(
            size_mb=128 if quick else 256,
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        # Compute benchmarks
        task = progress.add_task("Compute throughput...", total=3)

        if quick:
            compute_sizes = [2**i for i in range(22, 26)]  # 4M to 32M
        else:
            compute_sizes = [2**i for i in range(20, 28)]

        result.compute_results["fp32"] = benchmark_compute_fp32(
            sizes=compute_sizes,
            ops_per_thread=500 if quick else 1000,
            iterations=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.compute_results["fp16"] = benchmark_compute_fp16(
            sizes=compute_sizes,
            ops_per_thread=500 if quick else 1000,
            iterations=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        matmul_sizes = [512, 1024, 2048, 4096] if quick else [512, 1024, 2048, 4096, 8192]
        result.compute_results["matmul_fp16"] = benchmark_matmul_tflops(
            sizes=matmul_sizes,
            dtype=torch.float16,
            iterations=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        # Cache benchmarks
        task = progress.add_task("Cache hierarchy...", total=3)

        result.cache_results["l1"] = benchmark_l1_cache(
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.cache_results["l2"] = benchmark_l2_cache(
            iterations=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.cache_results["thrash"] = benchmark_cache_thrash(
            size_mb=128 if quick else 256,
            iterations=iterations,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        # Shared memory benchmarks
        task = progress.add_task("Shared memory...", total=2)

        result.shared_results["bandwidth"] = benchmark_shared_memory(
            size_mb=32 if quick else 64,
            iterations=500 if quick else 1000,
            runs=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        result.shared_results["bank_conflicts"] = benchmark_bank_conflicts(
            size_mb=32 if quick else 64,
            iterations=500 if quick else 1000,
            runs=iterations // 2,
            warmup=warmup,
            device=device,
        )
        progress.advance(task)

        # Roofline benchmarks
        task = progress.add_task("Roofline model...", total=1)

        intensities = [1, 4, 16, 64, 256, 1024] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        result.roofline_results["points"] = benchmark_roofline(
            size_mb=128 if quick else 256,
            intensities=intensities,
            iterations=iterations // 2,
            warmup=warmup,
            device=device,
        )
        result.roofline_results["plot"] = plot_roofline(result.roofline_results["points"])
        progress.advance(task)

    result.duration_seconds = time.time() - start_time

    if verbose:
        print_stress_report(result, console)

    return result


def print_stress_report(result: StressResult, console: Console | None = None):
    """Print a formatted report of stress test results."""
    if console is None:
        console = Console()

    console.print()
    console.print(Panel.fit(
        f"[bold blue]{result.gpu_name}[/bold blue]\n"
        f"Tested: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Duration: {result.duration_seconds:.1f}s",
        title="GPU Stress Test Report",
    ))

    # Memory bandwidth table
    if result.memory_results.get("bandwidth"):
        table = Table(title="Memory Bandwidth")
        table.add_column("Size", style="cyan")
        table.add_column("Bandwidth", style="green", justify="right")
        table.add_column("Efficiency", style="yellow", justify="right")

        for r in result.memory_results["bandwidth"][-5:]:  # Last 5 sizes
            table.add_row(
                f"{r.size_bytes // (1024*1024)} MB",
                f"{r.bandwidth_gbps:.1f} GB/s",
                f"{r.efficiency_pct:.1f}%",
            )
        console.print(table)

    # Coalescing comparison
    if result.memory_results.get("coalescing"):
        table = Table(title="Memory Coalescing Impact")
        table.add_column("Stride", style="cyan")
        table.add_column("Bandwidth", style="green", justify="right")
        table.add_column("Efficiency", style="yellow", justify="right")

        for r in result.memory_results["coalescing"]:
            color = "green" if r.efficiency_pct > 70 else "yellow" if r.efficiency_pct > 40 else "red"
            table.add_row(
                r.name,
                f"{r.bandwidth_gbps:.1f} GB/s",
                f"[{color}]{r.efficiency_pct:.1f}%[/{color}]",
            )
        console.print(table)

    # Compute throughput
    if result.compute_results.get("fp32"):
        table = Table(title="Compute Throughput (FP32)")
        table.add_column("Size", style="cyan")
        table.add_column("TFLOPS", style="green", justify="right")
        table.add_column("Efficiency", style="yellow", justify="right")

        for r in result.compute_results["fp32"][-4:]:
            table.add_row(
                f"{r.size // (1024*1024)}M threads",
                f"{r.tflops:.2f}",
                f"{r.efficiency_pct:.1f}%",
            )
        console.print(table)

    # Matmul TFLOPS
    if result.compute_results.get("matmul_fp16"):
        table = Table(title="Matrix Multiply (FP16)")
        table.add_column("Size", style="cyan")
        table.add_column("TFLOPS", style="green", justify="right")
        table.add_column("Efficiency", style="yellow", justify="right")

        for r in result.compute_results["matmul_fp16"]:
            table.add_row(
                f"{r.size}x{r.size}",
                f"{r.tflops:.1f}",
                f"{r.efficiency_pct:.1f}%",
            )
        console.print(table)

    # Cache hierarchy
    if result.cache_results.get("l2"):
        table = Table(title="Cache Hierarchy (L2 Boundary Test)")
        table.add_column("Working Set", style="cyan")
        table.add_column("Bandwidth", style="green", justify="right")
        table.add_column("Level", style="yellow")

        for r in result.cache_results["l2"]:
            if r.working_set_kb >= 1024:
                size_str = f"{r.working_set_kb // 1024} MB"
            else:
                size_str = f"{r.working_set_kb} KB"
            table.add_row(
                size_str,
                f"{r.bandwidth_gbps:.1f} GB/s",
                r.cache_level,
            )
        console.print(table)

    # Cache thrashing comparison
    if result.cache_results.get("thrash"):
        table = Table(title="Access Pattern Comparison")
        table.add_column("Pattern", style="cyan")
        table.add_column("Bandwidth", style="green", justify="right")

        for name, r in result.cache_results["thrash"].items():
            table.add_row(name, f"{r.bandwidth_gbps:.1f} GB/s")
        console.print(table)

    # Bank conflicts
    if result.shared_results.get("bank_conflicts"):
        table = Table(title="Shared Memory Bank Conflicts")
        table.add_column("Stride", style="cyan")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Relative", style="green", justify="right")

        baseline = result.shared_results["bank_conflicts"][0].duration_ms
        for r in result.shared_results["bank_conflicts"]:
            slowdown = r.duration_ms / baseline
            color = "green" if slowdown < 1.2 else "yellow" if slowdown < 2 else "red"
            table.add_row(
                r.name,
                f"{r.duration_ms:.3f} ms",
                f"[{color}]{slowdown:.2f}x[/{color}]",
            )
        console.print(table)

    # Roofline plot
    if result.roofline_results.get("plot"):
        console.print()
        console.print(Panel(result.roofline_results["plot"], title="Roofline Model"))

    # Summary
    console.print()
    peak_bandwidth = max(r.bandwidth_gbps for r in result.memory_results.get("bandwidth", [BandwidthDummy()]))
    peak_tflops = max(r.tflops for r in result.compute_results.get("matmul_fp16", [TflopsDummy()]))

    console.print(Panel.fit(
        f"[bold green]Peak Memory Bandwidth:[/bold green] {peak_bandwidth:.1f} GB/s\n"
        f"[bold green]Peak Compute (FP16 matmul):[/bold green] {peak_tflops:.1f} TFLOPS",
        title="Summary",
    ))


# Dummy classes for empty results
class BandwidthDummy:
    bandwidth_gbps = 0

class TflopsDummy:
    tflops = 0
