"""Nsight Compute (ncu) profiler backend."""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from io import StringIO
from typing import Iterator, Callable

from mage.profiler.backends.base import ProfilerBackend
from mage.profiler.models import KernelMetric


# Mapping from ncu metric names to KernelMetric fields
NCU_METRIC_MAP = {
    # Duration
    "gpu__time_duration.avg": "duration_us_raw",  # in nanoseconds, convert later

    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",

    # Memory bandwidth
    "dram__bytes.sum.per_second": "memory_throughput_raw",  # bytes/sec

    # Compute
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",

    # Cache
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",

    # Launch config
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_static": "static_shared_mem_bytes",
    "launch__shared_mem_per_block_dynamic": "dynamic_shared_mem_bytes",
    "launch__grid_size": "grid_size_raw",
    "launch__block_size": "block_size_raw",

    # Memory operations
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
}


class NcuBackend(ProfilerBackend):
    """Backend for NVIDIA Nsight Compute profiler."""

    name = "ncu"

    def __init__(self, metrics: list[str] | None = None):
        """Initialize ncu backend.

        Args:
            metrics: List of ncu metrics to collect. If None, uses defaults.
        """
        self.metrics = metrics or list(NCU_METRIC_MAP.keys())

    def get_command(self, script: str, args: list[str] | None = None) -> list[str]:
        """Build ncu profile command."""
        cmd = [
            "ncu",
            "--csv",
            "--target-processes", "all",
            "--set", "full",  # Collect comprehensive metrics
        ]

        # Add specific metrics if not using full set
        # for metric in self.metrics:
        #     cmd.extend(["--metrics", metric])

        cmd.extend(["python", script])
        if args:
            cmd.extend(args)
        return cmd

    def run(
        self,
        script: str,
        args: list[str] | None = None,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Run ncu and yield kernel metrics."""
        cmd = self.get_command(script, args)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate()

        # Parse CSV output
        if stdout.strip():
            yield from self._parse_csv_output(stdout, callback)

    def _parse_csv_output(
        self,
        csv_output: str,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Parse ncu CSV output into kernel metrics."""
        # ncu CSV has a header row and data rows
        # Each row represents one kernel invocation with all metrics

        lines = csv_output.strip().split("\n")
        if not lines:
            return

        # Find the header line (starts with "ID" or contains metric names)
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"') or line.startswith("ID,"):
                header_idx = i
                break

        if header_idx is None:
            return

        csv_text = "\n".join(lines[header_idx:])
        reader = csv.DictReader(StringIO(csv_text))

        current_kernel = None
        kernel_metrics: dict[str, dict] = {}

        for row in reader:
            kernel_name = row.get("Kernel Name", row.get("Name", "unknown"))
            if not kernel_name or kernel_name == "unknown":
                continue

            # ncu outputs one row per metric per kernel
            # Group metrics by kernel invocation
            kernel_id = row.get("ID", "0")
            key = f"{kernel_id}_{kernel_name}"

            if key not in kernel_metrics:
                kernel_metrics[key] = {
                    "kernel_name": kernel_name,
                    "timestamp": datetime.now(),
                }

            # Extract metric value
            metric_name = row.get("Metric Name", "")
            metric_value = row.get("Metric Value", row.get("Average", ""))

            if metric_name in NCU_METRIC_MAP:
                field = NCU_METRIC_MAP[metric_name]
                try:
                    kernel_metrics[key][field] = self._parse_value(metric_value)
                except (ValueError, TypeError):
                    pass

        # Convert to KernelMetric objects
        for data in kernel_metrics.values():
            metric = self._build_metric(data)
            if callback:
                callback(metric)
            yield metric

    def _parse_value(self, value: str) -> float | int | str:
        """Parse a metric value from string."""
        if not value:
            return None

        # Remove units and commas
        value = value.strip().replace(",", "")

        # Handle percentage
        if value.endswith("%"):
            return float(value[:-1])

        # Handle byte suffixes
        suffixes = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
        for suffix, multiplier in suffixes.items():
            if value.endswith(suffix):
                return float(value[:-1]) * multiplier

        # Try numeric conversion
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _build_metric(self, data: dict) -> KernelMetric:
        """Build a KernelMetric from parsed data."""
        # Convert raw values
        duration_us = data.get("duration_us_raw", 0)
        if isinstance(duration_us, (int, float)):
            duration_us = duration_us / 1000.0  # ns to us

        mem_throughput = data.get("memory_throughput_raw", None)
        if mem_throughput:
            mem_throughput = mem_throughput / 1e9  # bytes/s to GB/s

        occupancy = data.get("occupancy", None)
        if occupancy:
            occupancy = occupancy / 100.0  # percent to fraction

        # Parse grid/block size strings like "128,1,1"
        grid_size = self._parse_dim(data.get("grid_size_raw", "1,1,1"))
        block_size = self._parse_dim(data.get("block_size_raw", "1,1,1"))

        static_smem = data.get("static_shared_mem_bytes")
        dynamic_smem = data.get("dynamic_shared_mem_bytes")
        total_smem = None
        if static_smem is not None or dynamic_smem is not None:
            total_smem = (static_smem or 0) + (dynamic_smem or 0)

        return KernelMetric(
            kernel_name=data["kernel_name"],
            duration_us=duration_us or 0.0,
            timestamp=data.get("timestamp", datetime.now()),
            grid_size=grid_size,
            block_size=block_size,
            registers_per_thread=data.get("registers_per_thread"),
            shared_mem_bytes=total_smem,
            static_shared_mem_bytes=static_smem,
            dynamic_shared_mem_bytes=dynamic_smem,
            occupancy=occupancy,
            memory_throughput_gbps=mem_throughput,
            compute_throughput_pct=data.get("compute_throughput_pct"),
            l1_hit_rate=data.get("l1_hit_rate"),
            l2_hit_rate=data.get("l2_hit_rate"),
            dram_read_bytes=data.get("dram_read_bytes"),
            dram_write_bytes=data.get("dram_write_bytes"),
        )

    def _parse_dim(self, value: str | tuple | int) -> tuple[int, int, int]:
        """Parse a dimension string like '128,1,1' into a tuple."""
        if isinstance(value, tuple):
            return value
        if isinstance(value, int):
            return (value, 1, 1)
        if isinstance(value, str):
            parts = value.replace(" ", "").split(",")
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            elif len(parts) == 1:
                return (int(parts[0]), 1, 1)
        return (1, 1, 1)

    def parse_line(self, line: str) -> KernelMetric | None:
        """Parse a single CSV line (not typically used for ncu)."""
        return None
