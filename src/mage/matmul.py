"""High-performance matrix multiplication kernel with autotuning."""

from typing import Literal

import torch
import triton
import triton.language as tl

from mage.dtypes import validate_dtype

# Output dtype constants for kernel
OUTPUT_FP16 = 0
OUTPUT_BF16 = 1
OUTPUT_FP32 = 2


def get_cuda_autotune_configs():
    """Autotune configs optimized for NVIDIA GPUs."""
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    ]


@triton.autotune(configs=get_cuda_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Output dtype selector
    OUTPUT_DTYPE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Tiled matmul kernel with L2 cache optimization.

    Key optimizations:
    1. Grouped ordering: Programs are grouped to maximize L2 cache reuse
    2. Accumulate in fp32: Prevents precision loss in long dot products
    3. Tiled computation: Processes BLOCK_SIZE_M x BLOCK_SIZE_N output tiles
    """
    # Program ID and grouping for L2 cache optimization
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting positions
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to first blocks of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator in fp32 for precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks with boundary masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Accumulate
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert to output dtype based on constexpr selector
    # Use literal values: 0=fp16, 1=bf16, 2=fp32
    if OUTPUT_DTYPE == 0:
        c = accumulator.to(tl.float16)
    elif OUTPUT_DTYPE == 1:
        c = accumulator.to(tl.bfloat16)
    else:  # 2 = fp32
        c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _get_output_dtype_code(dtype: torch.dtype) -> int:
    """Map torch dtype to kernel output dtype constant."""
    if dtype == torch.float16:
        return OUTPUT_FP16
    elif dtype == torch.bfloat16:
        return OUTPUT_BF16
    else:
        return OUTPUT_FP32


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Matrix multiplication: C = A @ B

    Args:
        a: (M, K) tensor
        b: (K, N) tensor
        out_dtype: Output dtype. Defaults to input dtype.

    Returns:
        (M, N) tensor
    """
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} @ {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    # Validate and determine output dtype
    validate_dtype(a.dtype, "matmul input A")
    validate_dtype(b.dtype, "matmul input B")

    if out_dtype is None:
        out_dtype = a.dtype
    validate_dtype(out_dtype, "matmul output")

    M, K = a.shape
    K, N = b.shape

    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Get dtype code for kernel
    output_dtype_code = _get_output_dtype_code(out_dtype)

    # Grid: one program per output tile
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        OUTPUT_DTYPE=output_dtype_code,
    )
    return c
