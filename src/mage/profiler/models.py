"""Data models for GPU kernel profiling metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class KernelMetric:
    """Metrics for a single GPU kernel execution."""

    kernel_name: str
    duration_us: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Launch configuration
    grid_size: tuple[int, int, int] = (1, 1, 1)
    block_size: tuple[int, int, int] = (1, 1, 1)

    # Resource usage
    registers_per_thread: int | None = None
    shared_mem_bytes: int | None = None
    static_shared_mem_bytes: int | None = None
    dynamic_shared_mem_bytes: int | None = None

    # Performance metrics
    occupancy: float | None = None  # 0.0 - 1.0
    memory_throughput_gbps: float | None = None
    compute_throughput_pct: float | None = None

    # Cache metrics - L1
    l1_hit_rate: float | None = None
    l1_bytes_total: int | None = None
    l1_utilization: float | None = None  # % of peak

    # Cache metrics - L2
    l2_hit_rate: float | None = None
    l2_bytes_total: int | None = None
    l2_bytes_miss: int | None = None
    l2_utilization: float | None = None  # % of peak

    # Global memory (DRAM)
    dram_read_bytes: int | None = None
    dram_write_bytes: int | None = None
    dram_utilization: float | None = None  # % of peak

    # Memory efficiency
    global_load_efficiency: float | None = None  # ratio (ideal is 1.0)
    global_store_efficiency: float | None = None
    global_load_transactions: int | None = None
    global_store_transactions: int | None = None

    # Shared memory
    shared_utilization: float | None = None  # % of peak
    shared_bank_conflicts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["grid_size"] = ",".join(map(str, self.grid_size))
        d["block_size"] = ",".join(map(str, self.block_size))
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelMetric:
        """Create from dictionary."""
        d = d.copy()
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        if isinstance(d.get("grid_size"), str):
            d["grid_size"] = tuple(map(int, d["grid_size"].split(",")))
        if isinstance(d.get("block_size"), str):
            d["block_size"] = tuple(map(int, d["block_size"].split(",")))
        return cls(**d)

    @property
    def total_threads(self) -> int:
        """Total threads launched."""
        grid = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        block = self.block_size[0] * self.block_size[1] * self.block_size[2]
        return grid * block

    @property
    def arithmetic_intensity(self) -> float | None:
        """FLOPs per byte (if data available)."""
        if self.dram_read_bytes and self.dram_write_bytes:
            total_bytes = self.dram_read_bytes + self.dram_write_bytes
            if total_bytes > 0 and self.compute_throughput_pct:
                # Rough estimate - would need actual FLOP count for accuracy
                return self.compute_throughput_pct / (total_bytes / 1e9)
        return None


@dataclass
class ProfileSession:
    """A profiling session containing multiple kernel metrics."""

    command: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    metrics: list[KernelMetric] = field(default_factory=list)
    session_id: int | None = None

    # Session metadata
    device_name: str | None = None
    backend: str = "nsys"  # nsys or ncu

    def add_metric(self, metric: KernelMetric) -> None:
        """Add a kernel metric to the session."""
        self.metrics.append(metric)

    def finish(self) -> None:
        """Mark the session as finished."""
        self.end_time = datetime.now()

    @property
    def total_duration_us(self) -> float:
        """Total kernel execution time in microseconds."""
        return sum(m.duration_us for m in self.metrics)

    @property
    def kernel_count(self) -> int:
        """Number of kernel executions recorded."""
        return len(self.metrics)

    @property
    def unique_kernels(self) -> set[str]:
        """Set of unique kernel names."""
        return {m.kernel_name for m in self.metrics}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "command": self.command,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "device_name": self.device_name,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], metrics: list[KernelMetric] | None = None) -> ProfileSession:
        """Create from dictionary."""
        return cls(
            command=d["command"],
            start_time=datetime.fromisoformat(d["start_time"]),
            end_time=datetime.fromisoformat(d["end_time"]) if d.get("end_time") else None,
            device_name=d.get("device_name"),
            backend=d.get("backend", "nsys"),
            metrics=metrics or [],
            session_id=d.get("id"),
        )
