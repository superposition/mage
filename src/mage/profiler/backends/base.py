"""Abstract base class for profiler backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Callable

from mage.profiler.models import KernelMetric


class ProfilerBackend(ABC):
    """Abstract base class for GPU profiler backends (nsys, ncu, etc.)."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        script: str,
        args: list[str] | None = None,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Run the profiler on a script and yield metrics.

        Args:
            script: Path to the Python script to profile
            args: Additional arguments to pass to the script
            callback: Optional callback called for each metric (for live updates)

        Yields:
            KernelMetric objects as they are parsed from profiler output
        """
        pass

    @abstractmethod
    def parse_line(self, line: str) -> KernelMetric | None:
        """Parse a single line of profiler output.

        Args:
            line: A line from the profiler's stdout/stderr

        Returns:
            A KernelMetric if the line contains metric data, None otherwise
        """
        pass

    @abstractmethod
    def get_command(self, script: str, args: list[str] | None = None) -> list[str]:
        """Get the command to run the profiler.

        Args:
            script: Path to the Python script to profile
            args: Additional arguments to pass to the script

        Returns:
            List of command arguments for subprocess
        """
        pass

    def is_available(self) -> bool:
        """Check if this profiler is available on the system."""
        return self.find_executable() is not None

    def find_executable(self) -> str | None:
        """Find the profiler executable, checking common CUDA paths."""
        import shutil
        from pathlib import Path

        # Check PATH first
        exe = shutil.which(self.name)
        if exe:
            return exe

        # Check common CUDA installation paths
        cuda_paths = [
            Path("/usr/local/cuda/bin"),
            Path("/usr/local/cuda-12/bin"),
            Path("/usr/local/cuda-12.8/bin"),
            Path("/opt/nvidia/nsight-compute/"),
            Path.home() / ".local" / "cuda" / "bin",
        ]

        for cuda_path in cuda_paths:
            candidate = cuda_path / self.name
            if candidate.exists() and candidate.is_file():
                return str(candidate)

        return None
