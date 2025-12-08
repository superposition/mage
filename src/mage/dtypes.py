"""Dtype utilities and validation for Triton kernels."""

import torch

# FP8 dtype mapping (PyTorch 2.1+)
FP8_DTYPES: set[torch.dtype] = set()
if hasattr(torch, "float8_e5m2"):
    FP8_DTYPES.add(torch.float8_e5m2)
if hasattr(torch, "float8_e4m3fn"):
    FP8_DTYPES.add(torch.float8_e4m3fn)


def supports_fp8() -> bool:
    """Check if current GPU supports FP8 (compute capability >= 8.9).

    FP8 requires Ada Lovelace (RTX 40 series) or Hopper (H100) GPUs.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def validate_dtype(dtype: torch.dtype, operation: str = "operation") -> None:
    """Validate that dtype is supported on current device.

    Args:
        dtype: The dtype to validate
        operation: Name of the operation (for error messages)

    Raises:
        ValueError: If FP8 is requested but not supported
    """
    if dtype in FP8_DTYPES and not supports_fp8():
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        raise ValueError(
            f"FP8 dtype {dtype} requires compute capability >= 8.9 (Ada Lovelace/Hopper). "
            f"Current device: {device_name}"
        )


def get_accumulator_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Get the accumulator dtype for a given input dtype.

    Always returns float32 for numerical precision.
    """
    return torch.float32


# Supported dtypes for mage operations
SUPPORTED_DTYPES: set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
} | FP8_DTYPES
