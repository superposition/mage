"""The experimental Triton counterparts must meet the shared FP32 contracts."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BASE = Path(__file__).resolve().parents[1] / "examples" / "oxide"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load("experiment")
target = load("triton_target")
CASES = [(op, dims, False) for op, dims in experiment.DEFAULTS.items()]
CASES += [(op, dims, False) for op, dims in experiment.SMALL.items()]
CASES += [("layernorm", [2, 1], True), ("layernorm", [4, 768], True),
          ("neighbor", [7, 9, 0], False), ("neighbor", [1, 5, 9], False),
          ("matmul", [1, 1, 1], False)]


@pytest.mark.parametrize("op,dims,constant", CASES)
def test_full_output_matches_reference(tmp_path, op, dims, constant):
    experiment.generate(tmp_path, op, dims, constant=constant)
    _, reference = experiment.reference(tmp_path)
    _, implementation = target.implementation(tmp_path)
    expected = reference()
    actual = implementation()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
