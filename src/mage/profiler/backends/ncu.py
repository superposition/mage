"""Nsight Compute exports, with explicit units and missing-value handling."""
from __future__ import annotations
import csv
from io import StringIO
import math
import re

from .native import NativeBackend
from mage.profiler.models import KernelMetric

NCU_METRIC_MAP = {
    # Duration
    "gpu__time_duration.avg": "duration_us_raw",  # in nanoseconds, convert later

    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",

    # Memory bandwidth - Global/DRAM
    "dram__bytes.sum.per_second": "memory_throughput_raw",  # bytes/sec
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_utilization",

    # Memory - L2 Cache
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "lts__t_bytes.sum": "l2_bytes_total",
    "lts__t_bytes_lookup_miss.sum": "l2_bytes_miss",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_utilization",

    # Memory - L1/Texture Cache
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "l1tex__t_bytes.sum": "l1_bytes_total",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "l1_utilization",

    # Memory - Shared Memory
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed": "shared_utilization",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "shared_bank_conflicts",

    # Memory - Global Load/Store efficiency
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio": "global_load_efficiency",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio": "global_store_efficiency",

    # Memory transactions
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_load_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "global_store_transactions",

    # Compute
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",

    # Launch config
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_static": "static_shared_mem_bytes",
    "launch__shared_mem_per_block_dynamic": "dynamic_shared_mem_bytes",
    "launch__grid_size": "grid_size_raw",
    "launch__block_size": "block_size_raw",
}


_TIME_US = {"nsecond": .001, "ns": .001, "usecond": 1, "us": 1,
            "µs": 1, "msecond": 1000, "ms": 1000, "second": 1e6, "s": 1e6}
_BYTE_SCALE = {"byte": 1, "b": 1, "kbyte": 1e3, "kb": 1e3,
               "mbyte": 1e6, "mb": 1e6, "gbyte": 1e9, "gb": 1e9}
_FRACTIONS = {"occupancy"}
_BYTES_PER_SECTOR = {"global_load_efficiency", "global_store_efficiency"}
_INTEGER_FIELDS = {"registers_per_thread", "static_shared_mem_bytes", "dynamic_shared_mem_bytes",
                   "shared_bank_conflicts", "global_load_transactions", "global_store_transactions",
                   "l1_bytes_total", "l2_bytes_total", "l2_bytes_miss", "dram_read_bytes", "dram_write_bytes"}


class NcuBackend(NativeBackend):
    name = "ncu"

    def __init__(self, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics

    def get_exec_command(self, argv):
        executable = self.find_executable()
        if not executable:
            raise RuntimeError("ncu executable not found")
        if self.report_dir is None:
            self._prepare()
        command = [executable, "--csv", "--page", "raw", "--print-units", "base",
                   "--target-processes", "all", "--export", str(self.report_dir / "capture"),
                   "--log-file", str(self.report_dir / "metrics.csv")]
        command += ["--metrics", ",".join(self.metrics)] if self.metrics else ["--set", "full"]
        if self.capture_range == "cuda":
            command += ["--profile-from-start", "off"]
        if self.launch_count is not None:
            command += ["--launch-count", str(self.launch_count)]
        return command + list(argv)

    def read_metrics(self):
        path = self.report_dir / "metrics.csv"
        if not path.exists():
            raise RuntimeError(f"Nsight Compute CSV export is missing: {path}")
        yield from self._parse_csv_output(path.read_text(errors="replace"))

    @staticmethod
    def _parse_value(value):
        try:
            number = float(value.strip().replace(",", "").rstrip("%"))
            return number if math.isfinite(number) else None
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _parse_dim(value):
        # An aggregate block/thread count cannot recover the launch's three axes.
        parts = re.findall(r"\d+", value or "")
        return tuple(map(int, parts)) if len(parts) == 3 else None

    def _parse_csv_output(self, csv_output, callback=None):
        lines = csv_output.splitlines()
        header = next((i for i, line in enumerate(lines)
                       if line.startswith('"ID",') or line.startswith("ID,")), None)
        if header is None:
            raise RuntimeError("Nsight Compute output has no raw metric CSV header")
        reader = csv.DictReader(StringIO("\n".join(lines[header:])))
        if not {"Metric Name", "Metric Value", "Metric Unit", "Kernel Name"} <= set(reader.fieldnames or []):
            raise RuntimeError("Unsupported Nsight Compute CSV schema")
        groups = {}
        for row in reader:
            name = row.get("Kernel Name")
            if not name:
                continue
            key = tuple(row.get(k) for k in ("Process ID", "Context", "Stream", "ID", "Kernel Name"))
            data = groups.setdefault(key, {"kernel_name": name,
                "grid_size": self._parse_dim(row.get("Grid Size")),
                "block_size": self._parse_dim(row.get("Block Size"))})
            field = NCU_METRIC_MAP.get(row.get("Metric Name"))
            value = self._parse_value(row.get("Metric Value"))
            if field is None or value is None or field in {"grid_size_raw", "block_size_raw"}:
                continue
            unit = (row.get("Metric Unit") or "").strip().lower()
            if field == "duration_us_raw":
                if unit not in _TIME_US:
                    raise RuntimeError(f"Unsupported duration unit: {unit!r}")
                field, value = "duration_us", value * _TIME_US[unit]
            elif field == "memory_throughput_raw":
                scale = _BYTE_SCALE.get(unit.removesuffix("/second").removesuffix("/s"))
                if scale is None or "/" not in unit:
                    raise RuntimeError(f"Unsupported bandwidth unit: {unit!r}")
                field, value = "memory_throughput_gbps", value * scale / 1e9
            elif field in _FRACTIONS:
                if unit not in {"%", "pct"}:
                    raise RuntimeError(f"Unsupported percentage unit: {unit!r}")
                value /= 100
            elif field in _BYTES_PER_SECTOR:
                value /= 32  # SASS data bytes per 32-byte sector.
            elif field.endswith("_bytes") or field.endswith("_bytes_total") or field == "l2_bytes_miss":
                scale = _BYTE_SCALE.get(unit)
                if scale is None:
                    raise RuntimeError(f"Unsupported byte unit: {unit!r}")
                value *= scale
            data[field] = int(value) if field in _INTEGER_FIELDS else value
        for data in groups.values():
            if "duration_us" not in data:
                raise RuntimeError(f"Missing duration for {data['kernel_name']}; capture is incomplete")
            static, dynamic = data.get("static_shared_mem_bytes"), data.get("dynamic_shared_mem_bytes")
            if static is not None and dynamic is not None:
                data["shared_mem_bytes"] = static + dynamic
            metric = KernelMetric(**data)
            if callback:
                callback(metric)
            yield metric
