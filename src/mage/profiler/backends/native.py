"""Shared, shell-free process execution and retained artifacts for Nsight."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from .base import ProfilerBackend


class NativeBackend(ProfilerBackend):
    def __init__(self, *, capture_range="all", output_dir=None, launch_count=None):
        if capture_range not in {"all", "cuda"}:
            raise ValueError("capture_range must be all or cuda")
        if launch_count is not None and (self.name != "ncu" or launch_count <= 0):
            raise ValueError("launch_count must be positive and is only supported by ncu")
        self.capture_range = capture_range
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.launch_count = launch_count
        self.report_dir = None
        self._temporary = []

    def _prepare(self):
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(tempfile.mkdtemp(prefix=f"mage-{self.name}-", dir=self.output_dir))
        if self.output_dir is None:
            self._temporary.append(self.report_dir)
        return self.report_dir

    def get_command(self, script, args=None):
        return self.get_exec_command([sys.executable, str(Path(script).resolve()), *(args or [])])

    def run(self, script, args=None, callback=None):
        yield from self.run_command([sys.executable, str(Path(script).resolve()), *(args or [])], callback)

    def run_command(self, argv, callback=None):
        if not argv or not argv[0] or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Provide a nonempty executable argv after --")
        self._prepare()
        command = self.get_exec_command(argv)
        manifest = {"backend": self.name, "argv": argv, "profiler_argv": command,
                    "command": shlex.join(argv), "capture_range": self.capture_range,
                    "status": "running"}
        try:
            with (self.report_dir / "process.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                try:
                    returncode = process.wait()
                except BaseException:
                    if process.poll() is None:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                            process.wait()
                    raise
            manifest["returncode"] = returncode
            diagnostic = (self.report_dir / "process.log").read_text(errors="replace")
            csv_path = self.report_dir / "metrics.csv"
            if csv_path.exists():
                diagnostic += csv_path.read_text(errors="replace")
            if returncode or "ERR_NVGPUCTRPERM" in diagnostic:
                hint = ""
                if "ERR_NVGPUCTRPERM" in diagnostic:
                    hint = ("\nGPU performance counters are restricted. On WSL, enable access in "
                            "Windows NVIDIA Control Panel > Developer > Manage GPU Performance Counters. "
                            "See https://developer.nvidia.com/ERR_NVGPUCTRPERM")
                raise RuntimeError(f"{self.name} exited with code {returncode}. "
                                   f"Reports: {self.report_dir}\n{diagnostic[-4000:]}{hint}")
            metrics = list(self.read_metrics())
            if not metrics:
                raise RuntimeError(f"{self.name} captured no CUDA kernel launches. Check the target "
                                   f"and capture range. Reports: {self.report_dir}")
            rows = [metric.to_dict() for metric in metrics]
            (self.report_dir / "kernels.json").write_text(json.dumps(rows, indent=2) + "\n")
            with (self.report_dir / "kernels.csv").open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest.update(status="complete", kernel_count=len(metrics))
            for metric in metrics:
                if callback:
                    callback(metric)
                yield metric
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (self.report_dir / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def parse_line(self, line):
        # Structured exports are the only source of kernel metrics.
        return None

    def cleanup(self):
        for path in self._temporary:
            shutil.rmtree(path)
        self._temporary.clear()
