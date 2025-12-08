"""Triton-native profiler backend using torch.cuda.Event timing."""

from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Iterator, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import triton

from mage.profiler.backends.base import ProfilerBackend
from mage.profiler.models import KernelMetric


@dataclass
class KernelCall:
    """Record of a kernel call for profiling."""
    name: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    grid: tuple
    block_size: int | None = None
    num_warps: int | None = None
    shared_mem: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class TritonProfiler:
    """Context manager for profiling Triton kernels."""

    _instance: "TritonProfiler | None" = None
    _active: bool = False

    def __init__(self):
        self.calls: list[KernelCall] = []
        self._original_run = None

    @classmethod
    def get_instance(cls) -> "TritonProfiler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start profiling by patching triton.JITFunction.__call__."""
        if TritonProfiler._active:
            return

        self.calls.clear()
        self._patch_triton()
        TritonProfiler._active = True

    def stop(self):
        """Stop profiling and restore original behavior."""
        if not TritonProfiler._active:
            return

        self._unpatch_triton()
        TritonProfiler._active = False
        torch.cuda.synchronize()

    def _patch_triton(self):
        """Patch triton JIT function to record timings."""
        # Store original run method
        self._original_run = triton.JITFunction.run

        profiler = self

        @functools.wraps(self._original_run)
        def profiled_run(jit_fn, *args, grid, num_warps=None, num_stages=None,
                        num_ctas=1, enable_fp_fusion=True, extern_libs=None,
                        stream=None, warmup=False, **kwargs):
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Record start
            start_event.record()

            # Build kwargs for original run, excluding deprecated stream
            run_kwargs = {
                'grid': grid,
                'num_warps': num_warps,
                'num_stages': num_stages,
                'num_ctas': num_ctas,
                'enable_fp_fusion': enable_fp_fusion,
                'extern_libs': extern_libs,
                'warmup': warmup,
            }
            run_kwargs.update(kwargs)

            # Run the kernel
            result = profiler._original_run(jit_fn, *args, **run_kwargs)

            # Record end
            end_event.record()

            # Store the call info (don't sync yet)
            if not warmup:
                # Extract kernel info
                call = KernelCall(
                    name=jit_fn.fn.__name__,
                    start_event=start_event,
                    end_event=end_event,
                    grid=grid if isinstance(grid, tuple) else (grid,),
                    num_warps=num_warps,
                )
                profiler.calls.append(call)

            return result

        triton.JITFunction.run = profiled_run

    def _unpatch_triton(self):
        """Restore original triton behavior."""
        if self._original_run is not None:
            triton.JITFunction.run = self._original_run
            self._original_run = None

    def get_metrics(self) -> list[KernelMetric]:
        """Convert recorded calls to KernelMetric objects."""
        torch.cuda.synchronize()

        metrics = []
        for call in self.calls:
            # Get elapsed time in ms, convert to us
            duration_ms = call.start_event.elapsed_time(call.end_event)
            duration_us = duration_ms * 1000

            # Parse grid - handle callable grids by using stored value or default
            grid = call.grid
            if callable(grid):
                # Try to call it if it's a simple lambda
                try:
                    result = grid({})  # Try with empty dict for meta
                    if isinstance(result, (int, tuple)):
                        grid = result if isinstance(result, tuple) else (result,)
                    else:
                        grid = (1, 1, 1)
                except:
                    grid = (1, 1, 1)

            # Normalize grid to 3-tuple
            if isinstance(grid, int):
                grid = (grid, 1, 1)
            elif len(grid) == 1:
                grid = (grid[0], 1, 1)
            elif len(grid) == 2:
                grid = (grid[0], grid[1], 1)
            else:
                grid = tuple(grid[:3])

            # Estimate block size from num_warps (32 threads per warp)
            num_warps = call.num_warps or 4
            block_size = num_warps * 32
            block = (block_size, 1, 1)

            metrics.append(KernelMetric(
                kernel_name=call.name,
                duration_us=duration_us,
                timestamp=call.timestamp,
                grid_size=grid,
                block_size=block,
            ))

        return metrics


class TritonBackend(ProfilerBackend):
    """Backend using Triton's native profiling (no special permissions needed)."""

    name = "triton"

    def __init__(self):
        self._profiler: TritonProfiler | None = None

    def get_command(self, script: str, args: list[str] | None = None) -> list[str]:
        """Not used for Triton backend - we profile in-process."""
        return ["python", script] + (args or [])

    def is_available(self) -> bool:
        """Triton backend is always available if torch and triton are installed."""
        try:
            import torch
            import triton
            return torch.cuda.is_available()
        except ImportError:
            return False

    def run(
        self,
        script: str,
        args: list[str] | None = None,
        callback: Callable[[KernelMetric], None] | None = None,
    ) -> Iterator[KernelMetric]:
        """Run the script with Triton profiling enabled.

        Note: This runs the script in the current process with profiling hooks.
        """
        import subprocess
        import sys
        import runpy

        profiler = TritonProfiler.get_instance()

        # We need to run the script with our profiler active
        # This is done by importing and running with profiler context
        profiler.start()

        try:
            # Run the script
            old_argv = sys.argv
            sys.argv = [script] + (args or [])
            try:
                runpy.run_path(script, run_name="__main__")
            finally:
                sys.argv = old_argv

        finally:
            profiler.stop()

        # Get metrics and yield them
        metrics = profiler.get_metrics()
        for metric in metrics:
            if callback:
                callback(metric)
            yield metric

    def parse_line(self, line: str) -> KernelMetric | None:
        """Not used for Triton backend."""
        return None


@contextmanager
def profile_triton():
    """Context manager for profiling Triton kernels.

    Usage:
        with profile_triton() as profiler:
            # Run your Triton code
            result = my_kernel[grid](...)

        for metric in profiler.get_metrics():
            print(f"{metric.kernel_name}: {metric.duration_us:.2f} us")
    """
    profiler = TritonProfiler.get_instance()
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.stop()
