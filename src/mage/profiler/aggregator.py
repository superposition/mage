"""Metric aggregation with statistics, grouping, and time series."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any

from mage.profiler.models import KernelMetric


@dataclass
class RunningStats:
    """Online statistics using Welford's algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean
    min_val: float = float("inf")
    max_val: float = float("-inf")
    total: float = 0.0
    values: list[float] = field(default_factory=list)  # For percentiles

    def add(self, value: float) -> None:
        """Add a value to the running statistics."""
        if value is None:
            return

        self.count += 1
        self.total += value
        self.values.append(value)

        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance)

    def percentile(self, p: float) -> float | None:
        """Calculate percentile (0-100)."""
        if not self.values:
            return None
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def p50(self) -> float | None:
        """Median."""
        return self.percentile(50)

    @property
    def p95(self) -> float | None:
        """95th percentile."""
        return self.percentile(95)

    @property
    def p99(self) -> float | None:
        """99th percentile."""
        return self.percentile(99)

    def to_dict(self) -> dict[str, float | None]:
        """Export statistics as dictionary."""
        return {
            "count": self.count,
            "sum": self.total,
            "mean": self.mean if self.count > 0 else None,
            "std": self.std if self.count > 1 else None,
            "min": self.min_val if self.count > 0 else None,
            "max": self.max_val if self.count > 0 else None,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


class MetricAggregator:
    """Aggregates kernel metrics with statistics and grouping."""

    # Numeric fields that can be aggregated
    NUMERIC_FIELDS = [
        "duration_us",
        "occupancy",
        "memory_throughput_gbps",
        "compute_throughput_pct",
        "l1_hit_rate",
        "l2_hit_rate",
        "registers_per_thread",
        "shared_mem_bytes",
        "dram_read_bytes",
        "dram_write_bytes",
    ]

    def __init__(self):
        self._metrics: list[KernelMetric] = []
        self._stats: dict[str, RunningStats] = defaultdict(RunningStats)
        self._grouped_stats: dict[str, dict[str, RunningStats]] = defaultdict(
            lambda: defaultdict(RunningStats)
        )
        self._time_series: dict[str, list[tuple[datetime, KernelMetric]]] = defaultdict(list)

    def add(self, metric: KernelMetric) -> None:
        """Add a kernel metric to the aggregator."""
        self._metrics.append(metric)

        # Update global stats
        for field_name in self.NUMERIC_FIELDS:
            value = getattr(metric, field_name, None)
            if value is not None:
                self._stats[field_name].add(value)

        # Update per-kernel stats
        kernel_name = metric.kernel_name
        for field_name in self.NUMERIC_FIELDS:
            value = getattr(metric, field_name, None)
            if value is not None:
                self._grouped_stats[kernel_name][field_name].add(value)

        # Update time series
        self._time_series[kernel_name].append((metric.timestamp, metric))

    def stats(self, column: str) -> dict[str, float | None]:
        """Get aggregate statistics for a column.

        Args:
            column: Field name to get stats for

        Returns:
            Dictionary with count, sum, mean, std, min, max, p50, p95, p99
        """
        if column not in self._stats:
            return {}
        return self._stats[column].to_dict()

    def group_by(self, key: str = "kernel_name") -> dict[str, list[KernelMetric]]:
        """Group metrics by a field.

        Args:
            key: Field name to group by (default: kernel_name)

        Returns:
            Dictionary mapping key values to lists of metrics
        """
        groups: dict[str, list[KernelMetric]] = defaultdict(list)
        for metric in self._metrics:
            key_value = getattr(metric, key, "unknown")
            if isinstance(key_value, tuple):
                key_value = str(key_value)
            groups[str(key_value)].append(metric)
        return dict(groups)

    def group_stats(self, key: str = "kernel_name") -> dict[str, dict[str, dict]]:
        """Get statistics grouped by a field.

        Args:
            key: Field name to group by

        Returns:
            Nested dict: {group_value: {field_name: stats_dict}}
        """
        if key == "kernel_name":
            # Use pre-computed stats
            return {
                kernel: {
                    field: stats.to_dict()
                    for field, stats in field_stats.items()
                }
                for kernel, field_stats in self._grouped_stats.items()
            }

        # Compute on the fly for other keys
        groups = self.group_by(key)
        result = {}
        for group_value, metrics in groups.items():
            agg = MetricAggregator()
            for m in metrics:
                agg.add(m)
            result[group_value] = {
                field: agg.stats(field) for field in self.NUMERIC_FIELDS
            }
        return result

    def time_series(
        self,
        kernel_name: str,
        column: str = "duration_us",
    ) -> list[tuple[datetime, float]]:
        """Get time series data for a kernel and metric.

        Args:
            kernel_name: Kernel to get data for
            column: Metric field name

        Returns:
            List of (timestamp, value) tuples
        """
        result = []
        for timestamp, metric in self._time_series.get(kernel_name, []):
            value = getattr(metric, column, None)
            if value is not None:
                result.append((timestamp, value))
        return result

    def summary(self) -> dict[str, Any]:
        """Get a summary of all aggregated data."""
        return {
            "total_metrics": len(self._metrics),
            "unique_kernels": list(self._grouped_stats.keys()),
            "global_stats": {
                field: self._stats[field].to_dict()
                for field in self.NUMERIC_FIELDS
                if self._stats[field].count > 0
            },
        }

    @property
    def metrics(self) -> list[KernelMetric]:
        """All collected metrics."""
        return self._metrics

    @property
    def kernel_names(self) -> list[str]:
        """List of unique kernel names."""
        return list(self._grouped_stats.keys())

    def clear(self) -> None:
        """Clear all aggregated data."""
        self._metrics.clear()
        self._stats.clear()
        self._grouped_stats.clear()
        self._time_series.clear()

    def get_top_kernels(
        self,
        n: int = 10,
        by: str = "duration_us",
        stat: str = "sum",
    ) -> list[tuple[str, float]]:
        """Get top N kernels by a statistic.

        Args:
            n: Number of kernels to return
            by: Field to sort by
            stat: Statistic to use (sum, mean, max, count)

        Returns:
            List of (kernel_name, value) tuples
        """
        kernel_values = []
        for kernel, field_stats in self._grouped_stats.items():
            if by in field_stats:
                stats_dict = field_stats[by].to_dict()
                value = stats_dict.get(stat)
                if value is not None:
                    kernel_values.append((kernel, value))

        kernel_values.sort(key=lambda x: x[1], reverse=True)
        return kernel_values[:n]
