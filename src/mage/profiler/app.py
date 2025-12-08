"""Textual TUI app for GPU profiling with persistence and hot reload.

Run with hot reload:
    textual run --dev -c mage.profiler.app:ProfilerApp

Or directly:
    uv run python -m mage.profiler.app
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Static,
    Label,
    Button,
    ProgressBar,
)
from textual.reactive import reactive
from textual.message import Message

from mage.profiler.models import KernelMetric, ProfileSession
from mage.profiler.storage import ProfileDB
from mage.profiler.columns import COLUMNS, DEFAULT_COLUMNS, ColumnConfig
from mage.profiler.aggregator import MetricAggregator


class MetricRow:
    """Wrapper for a metric row in the table."""

    def __init__(self, metric: KernelMetric, columns: list[str]):
        self.metric = metric
        self.columns = columns

    def as_tuple(self) -> tuple:
        """Get values for table columns."""
        values = []
        for col_name in self.columns:
            col = COLUMNS.get(col_name)
            if col:
                values.append(col.format_value(self.metric))
            else:
                values.append("-")
        return tuple(values)


class StatusBar(Static):
    """Status bar showing session info."""

    metrics_count = reactive(0)
    kernel_count = reactive(0)
    elapsed = reactive(0.0)
    status = reactive("Ready")

    def render(self) -> str:
        return (
            f"[bold]{self.status}[/bold] │ "
            f"{self.metrics_count} metrics │ "
            f"{self.kernel_count} kernels │ "
            f"{self.elapsed:.1f}s"
        )


class SummaryPanel(Static):
    """Summary statistics panel."""

    def __init__(self, aggregator: MetricAggregator | None = None):
        super().__init__()
        self.aggregator = aggregator

    def update_stats(self, aggregator: MetricAggregator) -> None:
        self.aggregator = aggregator
        self.refresh()

    def render(self) -> str:
        if not self.aggregator or not self.aggregator.metrics:
            return "[dim]No data[/dim]"

        stats = self.aggregator.stats("duration_us")
        grouped = self.aggregator.group_by("kernel_name")

        lines = [
            "[bold]Summary[/bold]",
            f"Total: {stats['sum']:.1f} μs",
            f"Mean:  {stats['mean']:.1f} μs",
            f"Min:   {stats['min']:.1f} μs",
            f"Max:   {stats['max']:.1f} μs",
            "",
            "[bold]Top Kernels[/bold]",
        ]

        # Sort kernels by total time
        kernel_times = []
        for name, metrics in grouped.items():
            total = sum(m.duration_us for m in metrics)
            kernel_times.append((name, total, len(metrics)))

        kernel_times.sort(key=lambda x: x[1], reverse=True)

        for name, total, count in kernel_times[:5]:
            lines.append(f"  {name[:20]}: {total:.0f}μs ({count}x)")

        return "\n".join(lines)


class ProfilerApp(App):
    """Textual app for GPU kernel profiling."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: auto 1fr auto;
    }

    Header {
        column-span: 3;
    }

    #main-table {
        column-span: 2;
        row-span: 1;
        border: solid green;
    }

    #summary {
        column-span: 1;
        border: solid blue;
        padding: 1;
    }

    #status {
        column-span: 3;
        height: 1;
        background: $surface;
    }

    Footer {
        column-span: 3;
    }

    DataTable {
        height: 100%;
    }

    .column-header {
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "toggle_columns", "Columns"),
        Binding("s", "sort", "Sort"),
        Binding("g", "group", "Group"),
        Binding("h", "history", "History"),
        Binding("e", "export", "Export"),
    ]

    def __init__(
        self,
        columns: list[str] | None = None,
        db_path: str | None = None,
    ):
        super().__init__()
        self.column_config = ColumnConfig(columns)
        self.db = ProfileDB(db_path) if db_path else ProfileDB()
        self.aggregator = MetricAggregator()
        self.session: ProfileSession | None = None
        self._start_time: datetime | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="main-table")
        yield SummaryPanel(id="summary")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the table when app mounts."""
        table = self.query_one(DataTable)
        self._setup_columns(table)

        # Load last session if available
        self.load_last_session()

    def _setup_columns(self, table: DataTable) -> None:
        """Set up table columns."""
        table.clear(columns=True)
        for col_name in self.column_config.visible_names:
            col = COLUMNS.get(col_name)
            if col:
                table.add_column(col.header, key=col_name)

    def add_metric(self, metric: KernelMetric) -> None:
        """Add a metric to the display."""
        self.aggregator.add(metric)
        if self.session:
            self.session.add_metric(metric)

        table = self.query_one(DataTable)
        row = MetricRow(metric, self.column_config.visible_names)
        table.add_row(*row.as_tuple())

        # Update status
        status = self.query_one(StatusBar)
        status.metrics_count = len(self.aggregator.metrics)
        status.kernel_count = len(self.aggregator.group_by("kernel_name"))

        if self._start_time:
            status.elapsed = (datetime.now() - self._start_time).total_seconds()

        # Update summary
        summary = self.query_one(SummaryPanel)
        summary.update_stats(self.aggregator)

    def start_session(self, command: str, backend: str = "triton") -> None:
        """Start a new profiling session."""
        self.session = ProfileSession(
            command=command,
            backend=backend,
        )
        self._start_time = datetime.now()
        self.aggregator = MetricAggregator()

        # Clear table
        table = self.query_one(DataTable)
        table.clear()

        # Update status
        status = self.query_one(StatusBar)
        status.status = f"Profiling: {command}"
        status.metrics_count = 0
        status.kernel_count = 0
        status.elapsed = 0.0

    def finish_session(self) -> int | None:
        """Finish and save the current session."""
        if self.session:
            self.session.finish()
            session_id = self.db.save_session(self.session)

            status = self.query_one(StatusBar)
            status.status = f"Complete (saved id={session_id})"

            return session_id
        return None

    def load_last_session(self) -> None:
        """Load the most recent session from database."""
        sessions = self.db.get_sessions(limit=1)
        if sessions:
            session = sessions[0]
            self.session = session

            status = self.query_one(StatusBar)
            status.status = f"Loaded: {session.command}"
            status.metrics_count = len(session.metrics)
            status.kernel_count = len(session.unique_kernels)

            # Add metrics to table
            table = self.query_one(DataTable)
            for metric in session.metrics:
                self.aggregator.add(metric)
                row = MetricRow(metric, self.column_config.visible_names)
                table.add_row(*row.as_tuple())

            # Update summary
            summary = self.query_one(SummaryPanel)
            summary.update_stats(self.aggregator)

    def load_session(self, session_id: int) -> None:
        """Load a specific session by ID."""
        session = self.db.get_session(session_id)
        if session:
            self.session = session
            self.aggregator = MetricAggregator()

            # Clear and reload table
            table = self.query_one(DataTable)
            table.clear()

            for metric in session.metrics:
                self.aggregator.add(metric)
                row = MetricRow(metric, self.column_config.visible_names)
                table.add_row(*row.as_tuple())

            # Update UI
            status = self.query_one(StatusBar)
            status.status = f"Loaded: {session.command}"
            status.metrics_count = len(session.metrics)
            status.kernel_count = len(session.unique_kernels)

            summary = self.query_one(SummaryPanel)
            summary.update_stats(self.aggregator)

    def action_refresh(self) -> None:
        """Refresh the display."""
        if self.session:
            self.load_session(self.session.session_id or 0)

    def action_toggle_columns(self) -> None:
        """Toggle column visibility."""
        # TODO: Show column picker modal
        self.notify("Column picker coming soon!")

    def action_sort(self) -> None:
        """Change sort column."""
        table = self.query_one(DataTable)
        # Cycle through duration column sort
        self.column_config.cycle_sort("duration")
        self.notify(f"Sorting by duration ({'desc' if self.column_config.sort_reverse else 'asc'})")

    def action_group(self) -> None:
        """Toggle grouping."""
        # TODO: Implement grouping view
        self.notify("Grouping view coming soon!")

    def action_history(self) -> None:
        """Show session history."""
        sessions = self.db.get_sessions(limit=10)
        if sessions:
            lines = ["Recent Sessions:"]
            for s in sessions:
                lines.append(f"  [{s.session_id}] {s.command} ({len(s.metrics)} metrics)")
            self.notify("\n".join(lines))
        else:
            self.notify("No sessions found")

    def action_export(self) -> None:
        """Export current session."""
        if self.session and self.session.metrics:
            # Export to CSV
            path = Path(f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(path, "w") as f:
                headers = self.column_config.visible_names
                f.write(",".join(headers) + "\n")
                for metric in self.session.metrics:
                    row = MetricRow(metric, headers)
                    f.write(",".join(str(v) for v in row.as_tuple()) + "\n")
            self.notify(f"Exported to {path}")
        else:
            self.notify("No data to export")


def run_app():
    """Run the profiler app."""
    app = ProfilerApp()
    app.run()


if __name__ == "__main__":
    run_app()
