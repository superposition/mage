import pytest
import torch

from mage import add, matmul
from mage.dtypes import FP8_DTYPES, supports_fp8, validate_dtype


class TestDtypeSupport:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_matmul_dtypes(self, device, dtype):
        a = torch.randn(32, 32, device=device, dtype=dtype)
        b = torch.randn(32, 32, device=device, dtype=dtype)
        result = matmul(a, b)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_add_dtypes(self, device, dtype):
        x = torch.randn(100, device=device, dtype=dtype)
        y = torch.randn(100, device=device, dtype=dtype)
        result = add(x, y)
        assert result.dtype == dtype

    def test_matmul_output_dtype_override(self, device):
        a = torch.randn(32, 32, device=device, dtype=torch.float16)
        b = torch.randn(32, 32, device=device, dtype=torch.float16)
        result = matmul(a, b, out_dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_matmul_mixed_to_fp32(self, device):
        # Input in fp16, output in fp32 for higher precision
        a = torch.randn(64, 64, device=device, dtype=torch.float16)
        b = torch.randn(64, 64, device=device, dtype=torch.float16)
        result = matmul(a, b, out_dtype=torch.float32)
        expected = torch.matmul(a.float(), b.float())
        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_bfloat16(self, device):
        a = torch.randn(32, 32, device=device, dtype=torch.bfloat16)
        b = torch.randn(32, 32, device=device, dtype=torch.bfloat16)
        result = matmul(a, b)
        expected = torch.matmul(a, b)
        assert result.dtype == torch.bfloat16
        assert torch.allclose(result, expected, atol=1e-1, rtol=1e-1)


class TestFP8:
    @pytest.mark.skipif(not supports_fp8(), reason="FP8 not supported on this device")
    def test_fp8_device_check(self):
        # If we get here, FP8 is supported
        assert supports_fp8()

    @pytest.mark.skipif(supports_fp8(), reason="Test requires non-FP8 hardware")
    def test_fp8_error_on_unsupported(self):
        if not FP8_DTYPES:
            pytest.skip("FP8 dtypes not available in this PyTorch version")
        with pytest.raises(ValueError, match="compute capability"):
            validate_dtype(list(FP8_DTYPES)[0])


class TestDtypeValidation:
    def test_fp16_validation_passes(self):
        # Should not raise
        validate_dtype(torch.float16, "test")

    def test_bf16_validation_passes(self):
        # Should not raise
        validate_dtype(torch.bfloat16, "test")

    def test_fp32_validation_passes(self):
        # Should not raise
        validate_dtype(torch.float32, "test")
