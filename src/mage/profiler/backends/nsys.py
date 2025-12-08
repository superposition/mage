"""Nsight Systems (nsys) profiler backend."""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterator, Callable

from mage.profiler.backends.base import ProfilerBackend
from mage.profiler.models import KernelMetric


class NsysBackend(ProfilerBackend):
    """Backend for NVIDIA Nsight Systems profiler."""

    name = "nsys"

    def __init__(self):
        self._temp_dir = None

    def get_command(self, script: str, args: list[str] | None = None) -> list[str]:
        """Build nsys profile command."""
        nsys_path = self.find_executable()
        if not nsys_path:
            raise RuntimeError("nsys executable not found")

        self._temp_dir = tempfile.mkdtemp(prefix="mage_nsys_")
        output_path = Path(self._temp_dir) / "profile"

        cmd = [
            nsys_path, "profile",
            "--stats=true",
            "--force-overwrite=true",
            f"--output={output_path}",
            "--export=sqlite",
            "python", script,
        ]
        if args:
            cmd.extend(args)
        return cmd

    def run(
        self,
        script: str,
        args: list[str] | None = None,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Run nsys and yield kernel metrics."""
        cmd = self.get_command(script, args)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        in_cuda_kernel_stats = False
        header_line = None

        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)

            # Detect CUDA kernel statistics section
            if "CUDA Kernel Statistics:" in line:
                in_cuda_kernel_stats = True
                continue

            if in_cuda_kernel_stats:
                # End of section
                if line.startswith(" ") is False and line.strip() and ":" in line:
                    if "CUDA Kernel Statistics" not in line:
                        in_cuda_kernel_stats = False
                        continue

                # Parse CSV-like output
                if "Time (%)" in line and "Avg (ns)" in line:
                    header_line = line
                    continue

                if header_line and line.strip() and not line.startswith("-"):
                    metric = self._parse_kernel_stats_line(header_line, line)
                    if metric:
                        if callback:
                            callback(metric)
                        yield metric

        process.wait()

        # Also try to parse the sqlite export for more detailed data
        if self._temp_dir:
            sqlite_path = Path(self._temp_dir) / "profile.sqlite"
            if sqlite_path.exists():
                yield from self._parse_sqlite_report(sqlite_path, callback)

    def _parse_kernel_stats_line(self, header: str, line: str) -> KernelMetric | None:
        """Parse a line from the CUDA Kernel Statistics section."""
        try:
            # nsys stats output is space-aligned, parse by position
            # Example header: " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name"
            # The name is at the end and may contain spaces

            parts = line.split()
            if len(parts) < 9:
                return None

            # Name is everything after the 8th column
            # Find where numeric columns end
            name_parts = []
            numeric_count = 0
            for i, part in enumerate(parts):
                try:
                    float(part.replace(",", ""))
                    numeric_count += 1
                except ValueError:
                    if numeric_count >= 7:  # After all numeric columns
                        name_parts = parts[i:]
                        break

            if not name_parts:
                return None

            kernel_name = " ".join(name_parts)
            avg_ns = float(parts[3].replace(",", ""))

            return KernelMetric(
                kernel_name=kernel_name,
                duration_us=avg_ns / 1000.0,
                timestamp=datetime.now(),
            )
        except (ValueError, IndexError):
            return None

    def _parse_sqlite_report(
        self,
        sqlite_path: Path,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Parse detailed metrics from nsys sqlite export."""
        import sqlite3

        try:
            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row

            # Query CUDA kernel launches
            cursor = conn.execute("""
                SELECT
                    k.shortName as kernel_name,
                    k.end - k.start as duration_ns,
                    k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ,
                    k.registersPerThread,
                    k.staticSharedMemory,
                    k.dynamicSharedMemory
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                ORDER BY k.start
            """)

            for row in cursor:
                metric = KernelMetric(
                    kernel_name=row["kernel_name"] or "unknown",
                    duration_us=row["duration_ns"] / 1000.0,
                    timestamp=datetime.now(),
                    grid_size=(row["gridX"] or 1, row["gridY"] or 1, row["gridZ"] or 1),
                    block_size=(row["blockX"] or 1, row["blockY"] or 1, row["blockZ"] or 1),
                    registers_per_thread=row["registersPerThread"],
                    static_shared_mem_bytes=row["staticSharedMemory"],
                    dynamic_shared_mem_bytes=row["dynamicSharedMemory"],
                    shared_mem_bytes=(row["staticSharedMemory"] or 0) + (row["dynamicSharedMemory"] or 0),
                )
                if callback:
                    callback(metric)
                yield metric

            conn.close()
        except Exception:
            # sqlite export may not always be available
            pass

    def parse_line(self, line: str) -> KernelMetric | None:
        """Parse a single line (used for streaming)."""
        # This is mainly used by the stats parsing above
        return None

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir)
