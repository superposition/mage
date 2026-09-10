"""CPU-only regression tests for native capture boundaries and persisted evidence."""
import csv
from dataclasses import fields
from io import StringIO
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from mage.profiler.backends.native import NativeBackend
from mage.profiler.backends.ncu import NcuBackend
from mage.profiler.backends.nsys import NsysBackend
from mage.profiler.backends.triton_profiler import TritonBackend
from mage.profiler.models import KernelMetric, ProfileSession
from mage.profiler.storage import ProfileDB, SCHEMA


def ncu_csv(rows):
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["ID", "Process ID", "Context", "Stream", "Kernel Name", "Grid Size",
                     "Block Size", "Metric Name", "Metric Unit", "Metric Value"])
    for process, metric, unit, value in rows:
        writer.writerow([0, process, 1, 7, "kernel<float, 4>", "(2, 3, 1)", "(16, 16, 1)", metric, unit, value])
    return out.getvalue()


@pytest.mark.parametrize("unit,value", [("nsecond", "2e3"), ("usecond", "2"),
                                       ("msecond", "0.002"), ("second", "2e-6")])
def test_ncu_units_zero_missing_and_dimensions(unit, value):
    capture = ncu_csv([
        (42, "gpu__time_duration.avg", unit, value),
        (42, "sm__warps_active.avg.pct_of_peak_sustained_active", "%", "0"),
        (42, "dram__bytes.sum.per_second", "Gbyte/second", "12.5"),
        (42, "dram__bytes_read.sum", "Kbyte", "2"),
        (42, "lts__t_sector_hit_rate.pct", "%", "N/A"),
        (42, "launch__grid_size", "block", "6"),
    ])
    metric, = NcuBackend()._parse_csv_output(capture)
    assert metric.duration_us == pytest.approx(2)
    assert metric.occupancy == 0
    assert metric.memory_throughput_gbps == 12.5
    assert metric.dram_read_bytes == 2000
    assert metric.l2_hit_rate is None
    assert metric.grid_size == (2, 3, 1)
    assert metric.block_size == (16, 16, 1)
    assert metric.shared_mem_bytes is None


def test_ncu_process_ids_do_not_collide():
    metrics = list(NcuBackend()._parse_csv_output(ncu_csv([
        (42, "gpu__time_duration.avg", "usecond", "2"),
        (43, "gpu__time_duration.avg", "usecond", "5"),
    ])))
    assert [m.duration_us for m in metrics] == [2, 5]


def test_ncu_rejects_unknown_units_and_incomplete_capture():
    with pytest.raises(RuntimeError, match="duration unit"):
        list(NcuBackend()._parse_csv_output(ncu_csv([(1, "gpu__time_duration.avg", "cycles", "3")])))
    with pytest.raises(RuntimeError, match="Missing duration"):
        list(NcuBackend()._parse_csv_output(ncu_csv([(1, "dram__bytes_read.sum", "byte", "3")])))
    assert NcuBackend._parse_dim("512") is None


def test_nsys_resolves_string_ids_and_keeps_each_launch(tmp_path):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (31, 'tiled_matmul');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
        (start INTEGER, end INTEGER, shortName INTEGER, gridX INTEGER, gridY INTEGER,
         gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
         registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 3000, 31, 2, 3, 1, 16, 16, 1, 32, 2048, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4000, 7000, 31, 2, 3, 1, 16, 16, 1, 32, 2048, 0);
        """)
    metrics = list(NsysBackend()._parse_sqlite(path))
    assert [m.kernel_name for m in metrics] == ["tiled_matmul", "tiled_matmul"]
    assert [m.duration_us for m in metrics] == [2, 3]
    assert metrics[0].shared_mem_bytes == 2048


def test_legacy_database_migrates_and_all_fields_roundtrip(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO sessions (command, start_time) VALUES ('old', '2020-01-01T00:00:00')")
        conn.execute("INSERT INTO metrics (session_id,kernel_name,duration_us,timestamp) VALUES (1,'old',2,'2020-01-01T00:00:00')")
    db = ProfileDB(path)
    assert db.get_session(1).metrics[0].kernel_name == "old"
    kwargs = {field.name: (17 if str(field.type).startswith("int") else 1.5)
              for field in fields(KernelMetric) if field.name not in
              {"kernel_name", "timestamp", "duration_us", "grid_size", "block_size"}}
    metric = KernelMetric("all_fields", 2, grid_size=(2, 3, 1), block_size=(16, 16, 1), **kwargs)
    session = ProfileSession("native --name 'with spaces'", metrics=[metric])
    session.finish()
    saved = db.get_session(db.save_session(session))
    assert saved.metrics[0].to_dict() == metric.to_dict()
    assert ProfileDB(path).get_session(1).metrics[0].grid_size is None
    assert metric.arithmetic_intensity is None


class FakeNative(NativeBackend):
    name = "nsys"
    def get_exec_command(self, argv):
        return argv
    def read_metrics(self):
        yield KernelMetric("test", 2)


def test_argv_and_retained_artifacts(tmp_path):
    backend = FakeNative(output_dir=tmp_path)
    argv = [sys.executable, "-c", "import sys; print(sys.argv[1]); assert sys.argv[2] == ''", "a b;$(echo unsafe)", ""]
    seen = []
    assert len(list(backend.run_command(argv, seen.append))) == 1
    assert len(seen) == 1
    assert "a b;$(echo unsafe)" in (backend.report_dir / "process.log").read_text()
    assert json.loads((backend.report_dir / "capture.json").read_text())["argv"] == argv
    assert (backend.report_dir / "kernels.csv").exists()
    backend.cleanup()
    assert backend.report_dir.exists()


def test_failed_process_never_becomes_successful_session(tmp_path):
    backend = FakeNative(output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="code 7"):
        list(backend.run_command([sys.executable, "-c", "raise SystemExit(7)"]))
    manifest = json.loads((backend.report_dir / "capture.json").read_text())
    assert manifest["status"] == "failed"
    assert not (backend.report_dir / "kernels.json").exists()


def test_empty_capture_and_unsupported_backend(tmp_path, monkeypatch):
    backend = FakeNative(output_dir=tmp_path)
    monkeypatch.setattr(backend, "read_metrics", lambda: iter(()))
    with pytest.raises(RuntimeError, match="no CUDA kernel"):
        list(backend.run_command([sys.executable, "-c", "pass"]))
    with pytest.raises(ValueError, match="in process"):
        TritonBackend().run_command(["program"])
    with pytest.raises(ValueError, match="only supported by ncu"):
        NsysBackend(launch_count=1)


def test_cli_forwards_native_arguments(monkeypatch):
    from mage import cli
    received = {}
    def record(**kwargs):
        received.update(kwargs)
        return 0
    monkeypatch.setattr(cli, "profile", record)
    monkeypatch.setattr(sys, "argv", ["mage", "profile-exec", "--backend", "nsys",
                                     "--", "/path with spaces/program", "--backend", "target", ""])
    assert cli.main() == 0
    assert received["executable"] == ["/path with spaces/program", "--backend", "target", ""]
    assert received["backend"] == "nsys"
