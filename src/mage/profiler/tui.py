"""Rich TUI for live GPU profiling display."""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style

from mage.profiler.models import KernelMetric, ProfileSession
from mage.profiler.aggregator import MetricAggregator
from mage.profiler.columns import Column, ColumnConfig, COLUMNS


class ProfilerTUI:
    """Terminal UI for displaying profiling results."""

    def __init__(
        self,
        columns: list[str] | None = None,
        group_by: str | None = None,
        show_footer: bool = True,
    ):
        """Initialize the TUI.

        Args:
            columns: List of column names to display
            group_by: Field to group metrics by (None for no grouping)
            show_footer: Show aggregation footer row
        """
        self.console = Console()
        self.column_config = ColumnConfig(columns)
        self.group_by = group_by
        self.show_footer = show_footer

        self.aggregator = MetricAggregator()
        self.session: ProfileSession | None = None

        self._start_time: datetime | None = None
        self._running = False
        self._status = "Idle"

    def start_session(self, command: str, backend: str = "nsys") -> None:
        """Start a new profiling session."""
        self.session = ProfileSession(command=command, backend=backend)
        self._start_time = datetime.now()
        self._running = True
        self._status = "Profiling..."

    def add_metric(self, metric: KernelMetric) -> None:
        """Add a metric to the display."""
        self.aggregator.add(metric)
        if self.session:
            self.session.add_metric(metric)

    def finish(self) -> None:
        """Mark profiling as complete."""
        self._running = False
        self._status = "Complete"
        if self.session:
            self.session.finish()

    def _build_table(self) -> Table:
        """Build the metrics table."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=True,
        )

        # Add columns
        for col in self.column_config.visible_columns:
            table.add_column(
                col.header,
                justify=col.justify,
                width=col.width,
                no_wrap=True,
            )

        # Get metrics to display
        if self.group_by:
            # Show aggregated stats per group
            group_stats = self.aggregator.group_stats(self.group_by)
            for group_name, field_stats in sorted(group_stats.items()):
                row = self._build_group_row(group_name, field_stats)
                table.add_row(*row)
        else:
            # Show individual metrics
            metrics = self.aggregator.metrics
            # Sort if configured
            if self.column_config.sort_by:
                sort_col = COLUMNS.get(self.column_config.sort_by)
                if sort_col:
                    metrics = sorted(
                        metrics,
                        key=lambda m: sort_col.getter(m) or 0,
                        reverse=self.column_config.sort_reverse,
                    )

            for metric in metrics[-50:]:  # Show last 50
                row = self._build_metric_row(metric)
                table.add_row(*row)

        # Add footer with aggregates
        if self.show_footer and self.aggregator.metrics:
            table.add_section()
            footer_row = self._build_footer_row()
            table.add_row(*footer_row, style="bold")

        return table

    def _build_metric_row(self, metric: KernelMetric) -> list[Text]:
        """Build a table row for a single metric."""
        row = []
        for col in self.column_config.visible_columns:
            value_str = col.format_value(metric)
            color = col.get_color(metric)
            if color:
                row.append(Text(value_str, style=color))
            else:
                row.append(Text(value_str))
        return row

    def _build_group_row(
        self,
        group_name: str,
        field_stats: dict[str, dict],
    ) -> list[Text]:
        """Build a table row for grouped statistics."""
        row = []
        for col in self.column_config.visible_columns:
            if col.name == "kernel":
                row.append(Text(group_name))
            elif col.name in ("grid", "block", "threads"):
                # Can't aggregate these meaningfully
                row.append(Text("-"))
            else:
                # Map column name to field name
                field_map = {
                    "duration": "duration_us",
                    "occupancy": "occupancy",
                    "mem_bw": "memory_throughput_gbps",
                    "compute": "compute_throughput_pct",
                    "regs": "registers_per_thread",
                    "smem": "shared_mem_bytes",
                    "l1_hit": "l1_hit_rate",
                    "l2_hit": "l2_hit_rate",
                }
                field_name = field_map.get(col.name, col.name)
                stats = field_stats.get(field_name, {})
                mean = stats.get("mean")
                if mean is not None:
                    if col.name == "occupancy":
                        mean = mean * 100  # Convert to percentage
                    try:
                        value_str = f"{mean:{col.fmt}}" if col.fmt else str(mean)
                    except (ValueError, TypeError):
                        value_str = str(mean)
                    row.append(Text(value_str))
                else:
                    row.append(Text("-"))
        return row

    def _build_footer_row(self) -> list[Text]:
        """Build the aggregation footer row."""
        row = []
        for col in self.column_config.visible_columns:
            if col.name == "kernel":
                count = len(self.aggregator.kernel_names)
                row.append(Text(f"TOTAL ({count} kernels)", style="bold"))
            elif col.name == "duration":
                stats = self.aggregator.stats("duration_us")
                total = stats.get("sum", 0)
                row.append(Text(f"{total:.2f}", style="bold"))
            elif col.name == "occupancy":
                stats = self.aggregator.stats("occupancy")
                mean = stats.get("mean")
                if mean:
                    row.append(Text(f"{mean * 100:.1f}", style="bold"))
                else:
                    row.append(Text("-"))
            elif col.name == "mem_bw":
                stats = self.aggregator.stats("memory_throughput_gbps")
                mean = stats.get("mean")
                if mean:
                    row.append(Text(f"{mean:.1f}", style="bold"))
                else:
                    row.append(Text("-"))
            else:
                row.append(Text("-"))
        return row

    def _build_status_bar(self) -> Text:
        """Build the status bar."""
        parts = []

        # Elapsed time
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            parts.append(f"[cyan]{elapsed:.1f}s[/cyan]")

        # Metric count
        count = len(self.aggregator.metrics)
        parts.append(f"[green]{count}[/green] metrics")

        # Unique kernels
        kernels = len(self.aggregator.kernel_names)
        parts.append(f"[yellow]{kernels}[/yellow] kernels")

        # Status
        if self._running:
            parts.append("[bold blue]Profiling...[/bold blue]")
        else:
            parts.append(f"[bold]{self._status}[/bold]")

        # Help
        parts.append("[dim]q:quit  g:group  s:sort  c:columns[/dim]")

        return Text.from_markup(" │ ".join(parts))

    def render(self) -> Panel:
        """Render the full TUI."""
        table = self._build_table()
        status = self._build_status_bar()

        # Combine table and status
        layout = Layout()
        layout.split_column(
            Layout(table, name="table"),
            Layout(status, name="status", size=1),
        )

        return Panel(
            table,
            title=f"[bold]GPU Profiler[/bold] - {self.session.command if self.session else 'No session'}",
            subtitle=status,
            border_style="blue",
        )

    def run_live(
        self,
        metric_source: Callable[[], KernelMetric | None] | None = None,
        refresh_rate: float = 4.0,
    ) -> None:
        """Run the TUI with live updates.

        Args:
            metric_source: Optional callable that returns new metrics
            refresh_rate: Refreshes per second
        """
        with Live(
            self.render(),
            console=self.console,
            refresh_per_second=refresh_rate,
            screen=False,
        ) as live:
            try:
                while self._running:
                    if metric_source:
                        metric = metric_source()
                        if metric:
                            self.add_metric(metric)
                    live.update(self.render())
                    time.sleep(1 / refresh_rate)

                # Final update
                live.update(self.render())

            except KeyboardInterrupt:
                self._status = "Interrupted"
                self.finish()
                live.update(self.render())

    def print_final(self) -> None:
        """Print the final results (non-live mode)."""
        self.console.print(self.render())

    def print_summary(self) -> None:
        """Print a text summary of results."""
        self.console.print()
        self.console.print("[bold cyan]Summary[/bold cyan]")
        self.console.print("-" * 40)

        summary = self.aggregator.summary()
        self.console.print(f"Total metrics: {summary['total_metrics']}")
        self.console.print(f"Unique kernels: {len(summary['unique_kernels'])}")

        if "duration_us" in summary.get("global_stats", {}):
            duration_stats = summary["global_stats"]["duration_us"]
            self.console.print(f"\nDuration (μs):")
            self.console.print(f"  Total: {duration_stats['sum']:.2f}")
            self.console.print(f"  Mean:  {duration_stats['mean']:.2f}")
            self.console.print(f"  Min:   {duration_stats['min']:.2f}")
            self.console.print(f"  Max:   {duration_stats['max']:.2f}")

        # Top kernels by time
        self.console.print("\n[bold]Top kernels by total time:[/bold]")
        for kernel, total_time in self.aggregator.get_top_kernels(5, "duration_us", "sum"):
            self.console.print(f"  {kernel}: {total_time:.2f} μs")
