"""Native input validation before allocation; opt in by building the binary."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1] / "examples/oxide"
BINARY = BASE / "target/release/mage-oxide"
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="Build the cuda-oxide example first")


def run(directory):
    return subprocess.run([str(BINARY), str(directory)], capture_output=True, text=True)


def write_manifest(directory, op, dims, **extra):
    (directory / "input.json").write_text(json.dumps(
        {"op": op, "dims": dims, "warmup": 0, "iterations": 1, **extra}))


@pytest.mark.parametrize("dims", [[0, 1, 1], [1, 1], [2147483647, 2147483647, 1]])
def test_invalid_shapes(tmp_path, dims):
    write_manifest(tmp_path, "matmul", dims)
    result = run(tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "rust-output.bin").exists()


def test_truncated_input(tmp_path):
    write_manifest(tmp_path, "matmul", [1, 1, 1])
    (tmp_path / "a.bin").write_bytes(b"abc")
    assert "incorrect byte length" in run(tmp_path).stderr


@pytest.mark.parametrize("ptr,indices", [([0, 2, 1], [0]), ([0, 0, 1], [2]), ([1, 1, 1], [0])])
def test_invalid_csr(tmp_path, ptr, indices):
    write_manifest(tmp_path, "neighbor", [2, 1, 1])
    np.ones(2, dtype="<f4").tofile(tmp_path / "a.bin")
    np.ones(1, dtype="<f4").tofile(tmp_path / "b.bin")
    np.array(ptr, dtype="<u4").tofile(tmp_path / "rowptr.bin")
    np.array(indices, dtype="<u4").tofile(tmp_path / "indices.bin")
    assert "invalid CSR adjacency" in run(tmp_path).stderr


def test_nonfinite_inputs(tmp_path):
    write_manifest(tmp_path, "gelu", [1, 1])
    np.array([np.nan], dtype="<f4").tofile(tmp_path / "a.bin")
    assert "finite FP32" in run(tmp_path).stderr
