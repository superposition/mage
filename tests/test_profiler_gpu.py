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
