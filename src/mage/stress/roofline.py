"""Roofline model benchmarks.

The roofline model shows the relationship between:
- Arithmetic Intensity (AI): FLOPS / Bytes transferred
- Performance (GFLOPS)

Ridge point = Peak GFLOPS / Peak Bandwidth
For RTX 4090: 82.6 TFLOPS / 1008 GB/s ≈ 82 FLOPS/byte

Below ridge: Memory bound
Above ridge: Compute bound
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from dataclasses import dataclass


@triton.jit
def variable_intensity_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OPS_PER_ELEMENT: tl.constexpr,
):
    """Kernel with configurable arithmetic intensity.

    Performs OPS_PER_ELEMENT FMA operations per loaded element.
    AI = (2 * OPS_PER_ELEMENT) FLOPS / 8 bytes (load + store)
       = OPS_PER_ELEMENT / 4 FLOPS/byte
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load data (4 bytes)
    data = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # Variable compute intensity
    acc = data
    multiplier = 1.0001  # Slight variation to prevent optimization
    for _ in range(OPS_PER_ELEMENT):
        acc = acc * multiplier + data

    # Store result (4 bytes)
    tl.store(output_ptr + offs, acc, mask=mask)


@dataclass
class RooflinePoint:
    """A point on the roofline plot."""
    name: str
    arithmetic_intensity: float  # FLOPS/byte
    achieved_gflops: float
    achieved_bandwidth_gbps: float
    duration_ms: float
    bottleneck: str  # "memory" or "compute"

    def __repr__(self) -> str:
        return (
            f"{self.name}: AI={self.arithmetic_intensity:.1f} FLOP/B, "
            f"{self.achieved_gflops:.1f} GFLOPS ({self.bottleneck})"
        )


def benchmark_roofline(
    size_mb: int = 256,
    intensities: list[int] | None = None,
    warmup: int = 5,
    iterations: int = 50,
    device: str = "cuda",
    peak_gflops: float = 82600.0,  # RTX 4090 FP32
    peak_bandwidth_gbps: float = 1008.0,
) -> list[RooflinePoint]:
    """Generate roofline benchmark points at various arithmetic intensities.

    Args:
        size_mb: Data size in MB
        intensities: Operations per element (determines AI)
        warmup: Warmup iterations
        iterations: Timed iterations
        device: CUDA device
        peak_gflops: Peak compute in GFLOPS
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s

    Returns:
        List of RooflinePoint for each intensity
    """
    if intensities is None:
        # Range from very memory-bound to very compute-bound
        # AI = ops_per_element / 4 FLOPS/byte
        # Ridge point ≈ 82 FLOPS/byte = 328 ops_per_element
        intensities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    n_elements = (size_mb * 1024 * 1024) // 4
    BLOCK_SIZE = 256

    results = []
    ridge_point = peak_gflops / peak_bandwidth_gbps  # FLOPS/byte

    for ops_per_element in intensities:
        input_data = torch.randn(n_elements, device=device, dtype=torch.float32)
        output_data = torch.empty_like(input_data)

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Warmup
        for _ in range(warmup):
            variable_intensity_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_ELEMENT=ops_per_element,
            )
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            variable_intensity_kernel[grid](
                input_data, output_data, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                OPS_PER_ELEMENT=ops_per_element,
            )
        end.record()
        torch.cuda.synchronize()

        duration_ms = start.elapsed_time(end) / iterations

        # Calculate metrics
        # FLOPS: 2 per FMA (multiply + add) * ops_per_element per element
        total_flops = n_elements * ops_per_element * 2
        # Bytes: read (4) + write (4) per element
        total_bytes = n_elements * 8

        achieved_gflops = (total_flops / (duration_ms / 1000)) / 1e9
        achieved_bandwidth_gbps = (total_bytes / (duration_ms / 1000)) / 1e9
        arithmetic_intensity = total_flops / total_bytes  # FLOPS/byte

        # Determine bottleneck
        # Memory-bound: limited by bandwidth
        # Compute-bound: limited by peak FLOPS
        memory_limited_gflops = arithmetic_intensity * peak_bandwidth_gbps
        if memory_limited_gflops < peak_gflops:
            bottleneck = "memory"
        else:
            bottleneck = "compute"

        results.append(RooflinePoint(
            name=f"AI_{arithmetic_intensity:.1f}",
            arithmetic_intensity=arithmetic_intensity,
            achieved_gflops=achieved_gflops,
            achieved_bandwidth_gbps=achieved_bandwidth_gbps,
            duration_ms=duration_ms,
            bottleneck=bottleneck,
        ))

        del input_data, output_data
        torch.cuda.empty_cache()

    return results


def plot_roofline(
    points: list[RooflinePoint],
    peak_gflops: float = 82600.0,
    peak_bandwidth_gbps: float = 1008.0,
) -> str:
    """Generate ASCII roofline plot.

    Args:
        points: List of RooflinePoint from benchmark
        peak_gflops: Peak compute in GFLOPS
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s

    Returns:
        ASCII string representation of roofline plot
    """
    width = 60
    height = 20

    ridge_point = peak_gflops / peak_bandwidth_gbps

    # Find plot bounds
    max_ai = max(p.arithmetic_intensity for p in points)
    min_ai = min(p.arithmetic_intensity for p in points)
    max_gflops = max(p.achieved_gflops for p in points)

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw axes
    for y in range(height):
        grid[y][0] = '│'
    for x in range(width):
        grid[height - 1][x] = '─'
    grid[height - 1][0] = '└'

    # Draw roofline (memory-bound slope + compute-bound ceiling)
    for x in range(1, width):
        ai = min_ai * (max_ai / min_ai) ** (x / width)  # Log scale
        if ai < ridge_point:
            # Memory-bound: GFLOPS = AI * bandwidth
            roofline_gflops = ai * peak_bandwidth_gbps
        else:
            # Compute-bound: GFLOPS = peak
            roofline_gflops = peak_gflops

        y = int((1 - roofline_gflops / peak_gflops) * (height - 2))
        if 0 <= y < height - 1:
            grid[y][x] = '═'

    # Plot achieved points
    for p in points:
        x = int((p.arithmetic_intensity / max_ai) ** 0.5 * (width - 2)) + 1  # sqrt for log-ish scale
        y = int((1 - p.achieved_gflops / peak_gflops) * (height - 2))
        if 0 <= y < height - 1 and 0 < x < width:
            grid[y][x] = '●' if p.bottleneck == "memory" else '○'

    # Convert to string
    lines = [''.join(row) for row in grid]

    # Add labels
    header = f"Roofline Model (Peak: {peak_gflops/1000:.1f} TFLOPS, {peak_bandwidth_gbps:.0f} GB/s)"
    footer = f"● = memory-bound  ○ = compute-bound  Ridge point: {ridge_point:.1f} FLOP/B"

    return f"{header}\n{'─' * width}\n" + '\n'.join(lines) + f"\n{footer}"


def find_ridge_point(
    device: str = "cuda",
    peak_gflops: float = 82600.0,
    peak_bandwidth_gbps: float = 1008.0,
) -> dict:
    """Find the actual ridge point empirically.

    Binary search for the arithmetic intensity where
    the kernel transitions from memory-bound to compute-bound.
    """
    low = 1
    high = 4096

    theoretical_ridge = peak_gflops / peak_bandwidth_gbps

    results = {}
    results["theoretical_ridge"] = theoretical_ridge

    # Run a few targeted benchmarks
    test_intensities = [
        int(theoretical_ridge * 0.25),
        int(theoretical_ridge * 0.5),
        int(theoretical_ridge * 0.75),
        int(theoretical_ridge),
        int(theoretical_ridge * 1.5),
        int(theoretical_ridge * 2),
    ]

    points = benchmark_roofline(
        intensities=[max(1, i) for i in test_intensities],
        device=device,
        peak_gflops=peak_gflops,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
    )

    # Find transition point
    for i, p in enumerate(points):
        if p.bottleneck == "compute" and i > 0:
            results["empirical_ridge_low"] = points[i - 1].arithmetic_intensity
            results["empirical_ridge_high"] = p.arithmetic_intensity
            break

    results["points"] = points
    return results
