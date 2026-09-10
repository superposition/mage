"""GPU regression for repeated in-process profiling and exception cleanup."""
import pytest
import torch
import triton
from mage import add
from mage.profiler.backends.triton_profiler import profile_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_repeated_triton_contexts_restore_hook():
    x = torch.ones(1024, device="cuda")
    add(x, x)  # compile before timing
    original = triton.JITFunction.run
    for _ in range(2):
        with profile_triton() as profiler:
            result = add(x, x)
        assert triton.JITFunction.run is original
        torch.testing.assert_close(result, x * 2)
        assert len(profiler.get_metrics()) == 1
    with pytest.raises(RuntimeError, match="target failed"):
        with profile_triton():
            add(x, x)
            raise RuntimeError("target failed")
    assert triton.JITFunction.run is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_script_target_can_import_a_sibling_module(tmp_path):
    (tmp_path / "sibling.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "target_script.py").write_text(
        "from sibling import VALUE\n"
        "import torch\n"
        "from mage import add\n"
        "x = torch.ones(1024, device='cuda')\n"
        "add(x, x * VALUE)\n",
        encoding="utf-8",
    )
    from mage.profiler.backends import get_backend

    metrics = list(get_backend("triton").run(str(tmp_path / "target_script.py")))

    assert [metric.kernel_name for metric in metrics] == ["add_kernel"]
