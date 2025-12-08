"""SQLite persistence for profiling sessions and metrics."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from mage.profiler.models import KernelMetric, ProfileSession


DEFAULT_DB_PATH = Path.home() / ".mage" / "profiles.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    device_name TEXT,
    backend TEXT DEFAULT 'nsys'
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    kernel_name TEXT NOT NULL,
    duration_us REAL NOT NULL,
    timestamp TEXT NOT NULL,
    grid_size TEXT,
    block_size TEXT,
    registers_per_thread INTEGER,
    shared_mem_bytes INTEGER,
    static_shared_mem_bytes INTEGER,
    dynamic_shared_mem_bytes INTEGER,
    occupancy REAL,
    memory_throughput_gbps REAL,
    compute_throughput_pct REAL,
    l1_hit_rate REAL,
    l2_hit_rate REAL,
    dram_read_bytes INTEGER,
    dram_write_bytes INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_kernel ON metrics(kernel_name);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time);
"""


class ProfileDB:
    """SQLite database for storing profiling sessions and metrics."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_session(self, session: ProfileSession) -> int:
        """Save a profiling session and return its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (command, start_time, end_time, device_name, backend)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.command,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.device_name,
                    session.backend,
                ),
            )
            session_id = cursor.lastrowid

            # Save all metrics
            for metric in session.metrics:
                self._save_metric(conn, session_id, metric)

            return session_id

    def _save_metric(self, conn: sqlite3.Connection, session_id: int, metric: KernelMetric) -> int:
        """Save a single metric (internal, uses existing connection)."""
        cursor = conn.execute(
            """
            INSERT INTO metrics (
                session_id, kernel_name, duration_us, timestamp,
                grid_size, block_size, registers_per_thread, shared_mem_bytes,
                static_shared_mem_bytes, dynamic_shared_mem_bytes,
                occupancy, memory_throughput_gbps, compute_throughput_pct,
                l1_hit_rate, l2_hit_rate, dram_read_bytes, dram_write_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                metric.kernel_name,
                metric.duration_us,
                metric.timestamp.isoformat(),
                ",".join(map(str, metric.grid_size)),
                ",".join(map(str, metric.block_size)),
                metric.registers_per_thread,
                metric.shared_mem_bytes,
                metric.static_shared_mem_bytes,
                metric.dynamic_shared_mem_bytes,
                metric.occupancy,
                metric.memory_throughput_gbps,
                metric.compute_throughput_pct,
                metric.l1_hit_rate,
                metric.l2_hit_rate,
                metric.dram_read_bytes,
                metric.dram_write_bytes,
            ),
        )
        return cursor.lastrowid

    def save_metric(self, session_id: int, metric: KernelMetric) -> int:
        """Save a single metric to an existing session."""
        with self._connect() as conn:
            return self._save_metric(conn, session_id, metric)

    def get_session(self, session_id: int) -> ProfileSession | None:
        """Retrieve a session by ID with all its metrics."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if not row:
                return None

            metrics = self._get_session_metrics(conn, session_id)
            session = ProfileSession.from_dict(dict(row), metrics)
            session.session_id = session_id
            return session

    def _get_session_metrics(self, conn: sqlite3.Connection, session_id: int) -> list[KernelMetric]:
        """Get all metrics for a session."""
        rows = conn.execute(
            "SELECT * FROM metrics WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

        return [self._row_to_metric(row) for row in rows]

    def _row_to_metric(self, row: sqlite3.Row) -> KernelMetric:
        """Convert a database row to a KernelMetric."""
        d = dict(row)
        d.pop("id", None)
        d.pop("session_id", None)
        return KernelMetric.from_dict(d)

    def get_sessions(self, limit: int = 100, offset: int = 0) -> list[ProfileSession]:
        """Get recent sessions (without metrics for efficiency)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY start_time DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            return [ProfileSession.from_dict(dict(row)) for row in rows]

    def get_sessions_with_metrics(self, limit: int = 10) -> list[ProfileSession]:
        """Get recent sessions with all their metrics."""
        sessions = self.get_sessions(limit=limit)
        with self._connect() as conn:
            for session in sessions:
                if session.session_id:
                    session.metrics = self._get_session_metrics(conn, session.session_id)
        return sessions

    def time_series(
        self,
        kernel_name: str,
        metric: str = "duration_us",
        last_n_sessions: int = 50,
    ) -> list[tuple[datetime, float]]:
        """Get time series data for a specific kernel and metric.

        Args:
            kernel_name: Name of the kernel to query
            metric: Column name to retrieve (duration_us, occupancy, etc.)
            last_n_sessions: Number of recent sessions to include

        Returns:
            List of (timestamp, value) tuples
        """
        # Validate metric name to prevent SQL injection
        valid_metrics = {
            "duration_us", "occupancy", "memory_throughput_gbps",
            "compute_throughput_pct", "l1_hit_rate", "l2_hit_rate",
            "registers_per_thread", "shared_mem_bytes",
        }
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.timestamp, m.{metric}
                FROM metrics m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.kernel_name = ? AND m.{metric} IS NOT NULL
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                (kernel_name, last_n_sessions * 100),  # Approximate max metrics
            ).fetchall()

            return [
                (datetime.fromisoformat(row["timestamp"]), row[metric])
                for row in reversed(rows)  # Chronological order
            ]

    def get_kernel_stats(self, kernel_name: str, last_n_sessions: int = 10) -> dict:
        """Get aggregate statistics for a kernel across recent sessions."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(duration_us) as avg_duration,
                    MIN(duration_us) as min_duration,
                    MAX(duration_us) as max_duration,
                    AVG(occupancy) as avg_occupancy,
                    AVG(memory_throughput_gbps) as avg_mem_bw
                FROM metrics m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.kernel_name = ?
                AND s.id IN (
                    SELECT id FROM sessions ORDER BY start_time DESC LIMIT ?
                )
                """,
                (kernel_name, last_n_sessions),
            ).fetchone()

            return dict(row) if row else {}

    def list_kernels(self) -> list[str]:
        """Get list of all unique kernel names in the database."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT kernel_name FROM metrics ORDER BY kernel_name"
            ).fetchall()
            return [row["kernel_name"] for row in rows]

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its metrics."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return cursor.rowcount > 0

    def vacuum(self) -> None:
        """Reclaim disk space after deletions."""
        with self._connect() as conn:
            conn.execute("VACUUM")
