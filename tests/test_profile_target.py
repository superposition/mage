"""Tests for the `module:function` profiler target."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mage.profiler.targets import parse_target, write_driver

SAMPLE = (
    "from pathlib import Path\n"
    "LOG = Path(__file__).with_name('calls.log')\n"
    "def record(a, b=None):\n"
    "    with LOG.open('a') as f:\n"
    "        f.write(f'{a}:{b}\\n')\n"
)


def run_driver(driver: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(driver.parent, ignore_errors=True)


def test_parse_target_splits_module_and_attribute():
    assert parse_target("package.module:function") == ("package.module", "function")
    assert parse_target("module:Class.method") == ("module", "Class.method")


@pytest.mark.parametrize("spec", [
    "module",              # no separator
    ":function",           # no module
    "module:",             # no attribute
    "module:function:extra",
    "pack age:function",
    "module:1function",
])
def test_parse_target_rejects_unusable_specs(spec):
    with pytest.raises(ValueError):
        parse_target(spec)


def test_write_driver_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="--call-args"):
        write_driver("sample:fn", cwd=tmp_path, call_args="{")
    with pytest.raises(ValueError, match="--call-kwargs"):
        write_driver("sample:fn", cwd=tmp_path, call_kwargs="[]")
    with pytest.raises(ValueError, match="warmup"):
        write_driver("sample:fn", cwd=tmp_path, warmup=-1)
    with pytest.raises(ValueError, match="iterations"):
        write_driver("sample:fn", cwd=tmp_path, iterations=0)


def test_driver_calls_the_target_with_arguments(tmp_path):
    (tmp_path / "sample_target.py").write_text(SAMPLE, encoding="utf-8")
    driver = write_driver("sample_target:record", cwd=tmp_path, call_args="[1]", call_kwargs='{"b": 2}',
                          warmup=2, iterations=3)

    completed = run_driver(driver)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "calls.log").read_text().splitlines() == ["1:2"] * 5


def test_driver_reports_a_missing_attribute(tmp_path):
    (tmp_path / "sample_target.py").write_text(SAMPLE, encoding="utf-8")
    driver = write_driver("sample_target:absent", cwd=tmp_path, warmup=0, iterations=1)

    completed = run_driver(driver)

    assert completed.returncode != 0
    assert "absent" in completed.stderr


DISCARD_SAMPLE = (
    "from mage.profiler.backends.triton_profiler import TritonProfiler\n"
    "def step():\n"
    "    calls = TritonProfiler.get_instance().calls\n"
    "    calls.append(object())\n"
    "    print(len(calls), flush=True)\n"
)


def test_driver_discards_warmup_launches_only_when_asked(tmp_path):
    pytest.importorskip("triton")
    (tmp_path / "discard_target.py").write_text(DISCARD_SAMPLE, encoding="utf-8")

    discarded = run_driver(write_driver("discard_target:step", cwd=tmp_path, warmup=2, iterations=2,
                                        discard_warmup=True))
    assert discarded.returncode == 0, discarded.stderr
    assert discarded.stdout.split() == ["1", "2", "1", "2"]

    kept = run_driver(write_driver("discard_target:step", cwd=tmp_path, warmup=2, iterations=2))
    assert kept.returncode == 0, kept.stderr
    assert kept.stdout.split() == ["1", "2", "3", "4"]
