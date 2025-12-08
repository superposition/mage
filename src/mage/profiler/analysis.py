"""Memory analysis and roofline model for GPU kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from mage.profiler.models import KernelMetric
from mage.profiler.gpu_specs import GPUSpec, get_gpu_spec


@dataclass
class MemoryAnalysis:
    """Analysis results for a kernel's memory behavior."""

    kernel_name: str

    # Achieved metrics
    duration_us: float
    achieved_bandwidth_gbps: float | None
    achieved_occupancy: float | None

    # Memory efficiency
    load_efficiency: float | None  # 0-1
    store_efficiency: float | None  # 0-1

    # Cache behavior
    l1_hit_rate: float | None
    l2_hit_rate: float | None

    # Shared memory
    shared_mem_bytes: int | None
    bank_conflicts: int | None

    # Calculated metrics
    bytes_transferred: int | None
    arithmetic_intensity: float | None

    # Diagnosis
    bottleneck: Literal["memory", "compute", "latency", "unknown"]
    issues: list[str]
    recommendations: list[str]

    # Reference to GPU spec
    gpu_spec: GPUSpec | None = None

    @property
    def bandwidth_utilization(self) -> float | None:
        """Percentage of theoretical bandwidth achieved."""
        if self.achieved_bandwidth_gbps and self.gpu_spec:
            return self.achieved_bandwidth_gbps / self.gpu_spec.memory_bandwidth_gbps * 100
        return None

    @property
    def is_memory_bound(self) -> bool:
        """Whether kernel appears to be memory-bound."""
        return self.bottleneck == "memory"

    @property
    def is_compute_bound(self) -> bool:
        """Whether kernel appears to be compute-bound."""
        return self.bottleneck == "compute"


def analyze_kernel(
    metric: KernelMetric,
    gpu_spec: GPUSpec | None = None,
) -> MemoryAnalysis:
    """Analyze a kernel's memory behavior.

    Args:
        metric: Kernel metric to analyze
        gpu_spec: GPU specifications (auto-detected if None)

    Returns:
        MemoryAnalysis with diagnosis and recommendations
    """
    if gpu_spec is None:
        gpu_spec = get_gpu_spec()

    issues = []
    recommendations = []
    bottleneck: Literal["memory", "compute", "latency", "unknown"] = "unknown"

    # Calculate bytes transferred
    bytes_transferred = None
    if metric.dram_read_bytes or metric.dram_write_bytes:
        bytes_transferred = (metric.dram_read_bytes or 0) + (metric.dram_write_bytes or 0)

    # Calculate achieved bandwidth
    achieved_bw = metric.memory_throughput_gbps

    # Analyze occupancy
    if metric.occupancy is not None:
        occ_pct = metric.occupancy * 100 if metric.occupancy <= 1 else metric.occupancy
        if occ_pct < 25:
            issues.append(f"Very low occupancy ({occ_pct:.1f}%)")
            recommendations.append("Reduce registers/shared mem per thread, or increase block size")
        elif occ_pct < 50:
            issues.append(f"Low occupancy ({occ_pct:.1f}%)")
            recommendations.append("Consider adjusting block size or reducing resource usage")

    # Analyze memory coalescing
    if metric.global_load_efficiency is not None:
        eff = metric.global_load_efficiency
        if eff < 0.5:
            issues.append(f"Poor load coalescing ({eff*100:.1f}% efficiency)")
            recommendations.append("Restructure memory access for contiguous/coalesced reads")
        elif eff < 0.8:
            issues.append(f"Suboptimal load coalescing ({eff*100:.1f}% efficiency)")

    if metric.global_store_efficiency is not None:
        eff = metric.global_store_efficiency
        if eff < 0.5:
            issues.append(f"Poor store coalescing ({eff*100:.1f}% efficiency)")
            recommendations.append("Restructure memory access for contiguous/coalesced writes")

    # Analyze cache behavior
    if metric.l1_hit_rate is not None:
        if metric.l1_hit_rate < 50:
            issues.append(f"Low L1 cache hit rate ({metric.l1_hit_rate:.1f}%)")
            recommendations.append("Improve data locality or use shared memory for reused data")

    if metric.l2_hit_rate is not None:
        if metric.l2_hit_rate < 50:
            issues.append(f"Low L2 cache hit rate ({metric.l2_hit_rate:.1f}%)")

    # Analyze shared memory bank conflicts
    if metric.shared_bank_conflicts is not None and metric.shared_bank_conflicts > 0:
        issues.append(f"Shared memory bank conflicts: {metric.shared_bank_conflicts:,}")
        recommendations.append("Pad shared memory arrays to avoid bank conflicts")

    # Analyze bandwidth utilization
    if achieved_bw and gpu_spec:
        util_pct = achieved_bw / gpu_spec.memory_bandwidth_gbps * 100
        if util_pct > 70:
            bottleneck = "memory"
            issues.append(f"Memory bandwidth saturated ({util_pct:.1f}% of peak)")
            recommendations.append("Reduce data movement or increase arithmetic intensity")
        elif util_pct < 20 and metric.occupancy and metric.occupancy > 0.5:
            # High occupancy but low bandwidth - likely compute bound or latency bound
            if metric.compute_throughput_pct and metric.compute_throughput_pct > 70:
                bottleneck = "compute"
            else:
                bottleneck = "latency"
                issues.append("Low bandwidth utilization despite reasonable occupancy")
                recommendations.append("Check for latency-bound operations or warp divergence")

    # Determine bottleneck if not already set
    if bottleneck == "unknown":
        if achieved_bw and gpu_spec:
            util_pct = achieved_bw / gpu_spec.memory_bandwidth_gbps * 100
            if util_pct > 50:
                bottleneck = "memory"
            elif metric.compute_throughput_pct and metric.compute_throughput_pct > 50:
                bottleneck = "compute"
            else:
                bottleneck = "latency"

    # Estimate arithmetic intensity if we have the data
    ai = None
    if bytes_transferred and bytes_transferred > 0:
        # Rough estimate based on duration and compute capability
        # This is approximate without actual FLOP counts
        pass

    return MemoryAnalysis(
        kernel_name=metric.kernel_name,
        duration_us=metric.duration_us,
        achieved_bandwidth_gbps=achieved_bw,
        achieved_occupancy=metric.occupancy,
        load_efficiency=metric.global_load_efficiency,
        store_efficiency=metric.global_store_efficiency,
        l1_hit_rate=metric.l1_hit_rate,
        l2_hit_rate=metric.l2_hit_rate,
        shared_mem_bytes=metric.shared_mem_bytes,
        bank_conflicts=metric.shared_bank_conflicts,
        bytes_transferred=bytes_transferred,
        arithmetic_intensity=ai,
        bottleneck=bottleneck,
        issues=issues,
        recommendations=recommendations,
        gpu_spec=gpu_spec,
    )


def print_memory_report(
    metrics: list[KernelMetric],
    gpu_spec: GPUSpec | None = None,
    console: Console | None = None,
) -> None:
    """Print a comprehensive memory analysis report.

    Args:
        metrics: List of kernel metrics to analyze
        gpu_spec: GPU specifications (auto-detected if None)
        console: Rich console to print to
    """
    if console is None:
        console = Console()

    if gpu_spec is None:
        gpu_spec = get_gpu_spec()

    # Header with GPU info
    if gpu_spec:
        console.print(Panel(
            f"[bold]{gpu_spec.name}[/bold]\n"
            f"Memory: {gpu_spec.memory_gb}GB @ {gpu_spec.memory_bandwidth_gbps} GB/s\n"
            f"Compute: {gpu_spec.fp32_tflops} TFLOPS (FP32)\n"
            f"Balance Point: {gpu_spec.arithmetic_intensity_balance():.1f} FLOPs/byte",
            title="GPU Specifications",
            border_style="blue",
        ))
    else:
        console.print("[yellow]Warning: Could not detect GPU specifications[/yellow]")

    # Analyze each kernel
    analyses = [analyze_kernel(m, gpu_spec) for m in metrics]

    # Summary table
    table = Table(title="Memory Analysis Summary", show_header=True, header_style="bold cyan")
    table.add_column("Kernel", style="dim", width=30)
    table.add_column("Duration", justify="right")
    table.add_column("BW Util %", justify="right")
    table.add_column("Load Eff", justify="right")
    table.add_column("L1 Hit", justify="right")
    table.add_column("L2 Hit", justify="right")
    table.add_column("Bottleneck", justify="center")

    for analysis in analyses:
        # Color code bottleneck
        bn_style = {
            "memory": "red",
            "compute": "green",
            "latency": "yellow",
            "unknown": "dim",
        }.get(analysis.bottleneck, "dim")

        bw_util = f"{analysis.bandwidth_utilization:.1f}" if analysis.bandwidth_utilization else "-"
        load_eff = f"{analysis.load_efficiency*100:.0f}%" if analysis.load_efficiency else "-"
        l1 = f"{analysis.l1_hit_rate:.0f}%" if analysis.l1_hit_rate else "-"
        l2 = f"{analysis.l2_hit_rate:.0f}%" if analysis.l2_hit_rate else "-"

        table.add_row(
            analysis.kernel_name[:30],
            f"{analysis.duration_us:.2f}μs",
            bw_util,
            load_eff,
            l1,
            l2,
            Text(analysis.bottleneck.upper(), style=bn_style),
        )

    console.print(table)

    # Detailed issues and recommendations
    all_issues = []
    all_recommendations = []

    for analysis in analyses:
        for issue in analysis.issues:
            all_issues.append(f"[{analysis.kernel_name}] {issue}")
        for rec in analysis.recommendations:
            all_recommendations.append(f"[{analysis.kernel_name}] {rec}")

    if all_issues:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for issue in all_issues:
            console.print(f"  • {issue}")

    if all_recommendations:
        console.print("\n[bold green]Recommendations:[/bold green]")
        # Deduplicate similar recommendations
        seen = set()
        for rec in all_recommendations:
            # Extract just the recommendation part
            rec_text = rec.split("] ", 1)[-1] if "] " in rec else rec
            if rec_text not in seen:
                seen.add(rec_text)
                console.print(f"  • {rec}")

    # Roofline summary
    if gpu_spec:
        console.print("\n[bold]Roofline Analysis:[/bold]")
        balance = gpu_spec.arithmetic_intensity_balance()
        console.print(f"  • Ridge point: {balance:.1f} FLOPs/byte")
        console.print(f"  • Below {balance:.0f} FLOPs/byte → Memory-bound")
        console.print(f"  • Above {balance:.0f} FLOPs/byte → Compute-bound")

        # Categorize kernels
        mem_bound = [a for a in analyses if a.bottleneck == "memory"]
        compute_bound = [a for a in analyses if a.bottleneck == "compute"]
        latency_bound = [a for a in analyses if a.bottleneck == "latency"]

        if mem_bound:
            console.print(f"\n  Memory-bound kernels ({len(mem_bound)}):")
            for a in mem_bound[:5]:
                console.print(f"    - {a.kernel_name}")

        if compute_bound:
            console.print(f"\n  Compute-bound kernels ({len(compute_bound)}):")
            for a in compute_bound[:5]:
                console.print(f"    - {a.kernel_name}")

        if latency_bound:
            console.print(f"\n  Latency-bound kernels ({len(latency_bound)}):")
            for a in latency_bound[:5]:
                console.print(f"    - {a.kernel_name}")
