"""Nsight Systems: one metric per CUPTI kernel launch, never summary duplicates."""
from pathlib import Path
import sqlite3

from .native import NativeBackend
from mage.profiler.models import KernelMetric


class NsysBackend(NativeBackend):
    name = "nsys"

    def get_exec_command(self, argv):
        executable = self.find_executable()
        if not executable:
            raise RuntimeError("nsys executable not found")
        if self.report_dir is None:
            self._prepare()
        command = [executable, "profile", "--trace=cuda,nvtx", "--sample=none",
                   "--cpuctxsw=none", "--stats=false", "--export=sqlite",
                   f"--output={self.report_dir / 'capture'}"]
        if self.capture_range == "cuda":
            command += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
        return command + list(argv)

    def read_metrics(self):
        yield from self._parse_sqlite(self.report_dir / "capture.sqlite")

    def _parse_sqlite(self, path):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Nsight Systems SQLite export is missing: {path}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            table = "CUPTI_ACTIVITY_KIND_KERNEL"
            if table not in tables:
                return
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if not {"start", "end"} <= columns:
                raise RuntimeError("Unsupported Nsight Systems kernel schema (missing start/end)")
            names = dict(conn.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY start"):
                data = dict(row)
                name = next((data[key] for key in ("demangledName", "shortName", "name")
                             if data.get(key) is not None), None)
                if isinstance(name, int):
                    name = names.get(name)
                if not isinstance(name, str) or not name:
                    raise RuntimeError("Cannot resolve kernel name in Nsight Systems StringIds")
                def dim(prefix):
                    values = tuple(data.get(prefix + axis) for axis in "XYZ")
                    return values if all(v is not None for v in values) else None
                static, dynamic = data.get("staticSharedMemory"), data.get("dynamicSharedMemory")
                if data["end"] < data["start"]:
                    raise RuntimeError("Invalid negative CUDA kernel duration in Systems export")
                yield KernelMetric(
                    kernel_name=name, duration_us=(data["end"] - data["start"]) / 1000,
                    grid_size=dim("grid"), block_size=dim("block"),
                    registers_per_thread=data.get("registersPerThread"),
                    static_shared_mem_bytes=static, dynamic_shared_mem_bytes=dynamic,
                    shared_mem_bytes=static + dynamic if static is not None and dynamic is not None else None)
