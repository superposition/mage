"""Column definitions and configuration for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from mage.profiler.models import KernelMetric


@dataclass
class Column:
    """Definition of a displayable column."""

    name: str
    header: str
    getter: Callable[[KernelMetric], Any]
    fmt: str = ""
    width: int = 12
    visible: bool = True
    justify: str = "right"  # left, center, right
    color: str | None = None
    # Color thresholds: (low_threshold, high_threshold, low_color, mid_color, high_color)
    color_thresholds: tuple[float, float, str, str, str] | None = None

    def format_value(self, metric: KernelMetric) -> str:
        """Format a metric value for display."""
        value = self.getter(metric)
        if value is None:
            return "-"

        if self.fmt:
            try:
                return f"{value:{self.fmt}}"
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    def get_color(self, metric: KernelMetric) -> str | None:
        """Get color for a value based on thresholds."""
        if self.color:
            return self.color

        if self.color_thresholds:
            value = self.getter(metric)
            if value is None:
                return None
            low, high, low_color, mid_color, high_color = self.color_thresholds
            if value < low:
                return low_color
            elif value > high:
                return high_color
            return mid_color

        return None


def _format_grid(metric: KernelMetric) -> str:
    """Format grid size as string."""
    g = metric.grid_size
    if g[1] == 1 and g[2] == 1:
        return str(g[0])
    elif g[2] == 1:
        return f"{g[0]}x{g[1]}"
    return f"{g[0]}x{g[1]}x{g[2]}"


def _format_block(metric: KernelMetric) -> str:
    """Format block size as string."""
    b = metric.block_size
    if b[1] == 1 and b[2] == 1:
        return str(b[0])
    elif b[2] == 1:
        return f"{b[0]}x{b[1]}"
    return f"{b[0]}x{b[1]}x{b[2]}"


def _format_bytes(value: int | None) -> str:
    """Format bytes with SI suffix."""
    if value is None:
        return "-"
    for suffix in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024:
            return f"{value:.1f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
        value /= 1024
    return f"{value:.1f}PB"


# Default column definitions
COLUMNS: dict[str, Column] = {
    "kernel": Column(
        name="kernel",
        header="Kernel",
        getter=lambda m: m.kernel_name,
        width=35,
        justify="left",
    ),
    "duration": Column(
        name="duration",
        header="Duration (μs)",
        getter=lambda m: m.duration_us,
        fmt=".2f",
        width=14,
    ),
    "occupancy": Column(
        name="occupancy",
        header="Occupancy",
        getter=lambda m: m.occupancy * 100 if m.occupancy else None,
        fmt=".1f",
        width=10,
        # Red < 50%, Yellow 50-75%, Green > 75%
        color_thresholds=(50.0, 75.0, "red", "yellow", "green"),
    ),
    "mem_bw": Column(
        name="mem_bw",
        header="Mem BW (GB/s)",
        getter=lambda m: m.memory_throughput_gbps,
        fmt=".1f",
        width=14,
    ),
    "compute": Column(
        name="compute",
        header="Compute %",
        getter=lambda m: m.compute_throughput_pct,
        fmt=".1f",
        width=10,
    ),
    "grid": Column(
        name="grid",
        header="Grid",
        getter=_format_grid,
        width=12,
        justify="left",
    ),
    "block": Column(
        name="block",
        header="Block",
        getter=_format_block,
        width=12,
        justify="left",
    ),
    "regs": Column(
        name="regs",
        header="Regs",
        getter=lambda m: m.registers_per_thread,
        width=6,
    ),
    "smem": Column(
        name="smem",
        header="Shared Mem",
        getter=lambda m: _format_bytes(m.shared_mem_bytes) if m.shared_mem_bytes else None,
        width=11,
    ),
    "l1_hit": Column(
        name="l1_hit",
        header="L1 Hit %",
        getter=lambda m: m.l1_hit_rate,
        fmt=".1f",
        width=9,
        color_thresholds=(50.0, 80.0, "red", "yellow", "green"),
    ),
    "l2_hit": Column(
        name="l2_hit",
        header="L2 Hit %",
        getter=lambda m: m.l2_hit_rate,
        fmt=".1f",
        width=9,
        color_thresholds=(50.0, 80.0, "red", "yellow", "green"),
    ),
    "dram_read": Column(
        name="dram_read",
        header="DRAM Read",
        getter=lambda m: _format_bytes(m.dram_read_bytes) if m.dram_read_bytes else None,
        width=11,
    ),
    "dram_write": Column(
        name="dram_write",
        header="DRAM Write",
        getter=lambda m: _format_bytes(m.dram_write_bytes) if m.dram_write_bytes else None,
        width=11,
    ),
    "threads": Column(
        name="threads",
        header="Threads",
        getter=lambda m: m.total_threads,
        fmt=",d",
        width=12,
    ),
}

# Default columns to show
DEFAULT_COLUMNS = ["kernel", "duration", "occupancy", "mem_bw", "grid", "block"]


class ColumnConfig:
    """Manages column visibility and ordering."""

    def __init__(self, columns: list[str] | None = None):
        """Initialize with specified columns or defaults.

        Args:
            columns: List of column names to show, or None for defaults
        """
        self._visible = columns or DEFAULT_COLUMNS.copy()
        self._sort_by: str | None = "duration"
        self._sort_reverse: bool = True

    @property
    def visible_columns(self) -> list[Column]:
        """Get list of visible Column objects in order."""
        return [COLUMNS[name] for name in self._visible if name in COLUMNS]

    @property
    def visible_names(self) -> list[str]:
        """Get list of visible column names."""
        return self._visible.copy()

    def show(self, name: str) -> None:
        """Show a column."""
        if name in COLUMNS and name not in self._visible:
            self._visible.append(name)

    def hide(self, name: str) -> None:
        """Hide a column."""
        if name in self._visible:
            self._visible.remove(name)

    def toggle(self, name: str) -> bool:
        """Toggle column visibility. Returns new visibility state."""
        if name in self._visible:
            self.hide(name)
            return False
        else:
            self.show(name)
            return True

    def set_columns(self, names: list[str]) -> None:
        """Set visible columns."""
        self._visible = [n for n in names if n in COLUMNS]

    def move_left(self, name: str) -> None:
        """Move a column left in the order."""
        if name in self._visible:
            idx = self._visible.index(name)
            if idx > 0:
                self._visible[idx], self._visible[idx - 1] = (
                    self._visible[idx - 1],
                    self._visible[idx],
                )

    def move_right(self, name: str) -> None:
        """Move a column right in the order."""
        if name in self._visible:
            idx = self._visible.index(name)
            if idx < len(self._visible) - 1:
                self._visible[idx], self._visible[idx + 1] = (
                    self._visible[idx + 1],
                    self._visible[idx],
                )

    @property
    def sort_by(self) -> str | None:
        """Current sort column."""
        return self._sort_by

    @property
    def sort_reverse(self) -> bool:
        """Sort in descending order."""
        return self._sort_reverse

    def set_sort(self, column: str, reverse: bool = True) -> None:
        """Set sort column and direction."""
        if column in COLUMNS:
            self._sort_by = column
            self._sort_reverse = reverse

    def cycle_sort(self, column: str) -> None:
        """Cycle sort: none -> desc -> asc -> none."""
        if self._sort_by != column:
            self._sort_by = column
            self._sort_reverse = True
        elif self._sort_reverse:
            self._sort_reverse = False
        else:
            self._sort_by = None

    @staticmethod
    def available_columns() -> list[str]:
        """Get list of all available column names."""
        return list(COLUMNS.keys())
