"""Batched small matrix multiplication for robotics applications.

Optimized for scenarios with many small matrix multiplications:
- Robot dynamics: (16x16) @ (16x16) batched over 1000+ instances
- Multi-agent simulations: small state/action matrices
- Jacobian computations: many small matrices in parallel

cuBLAS is inefficient for small matrices due to kernel launch overhead.
This implementation uses a fused kernel for better throughput.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _batch_matmul_kernel(
    A,  # (batch, M, K)
    B,  # (batch, K, N)
    C,  # (batch, M, N)
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matrix multiplication kernel.

    Each program handles one (batch, m_tile, n_tile) combination.
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for this batch
    a_batch_ptr = A + pid_batch * stride_ab
    b_batch_ptr = B + pid_batch * stride_bb
    c_batch_ptr = C + pid_batch * stride_cb

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = a_batch_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: (BLOCK_K, BLOCK_N)
        b_ptrs = b_batch_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc = tl.dot(a, b, acc)

    # Store result
    c_ptrs = c_batch_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def batch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched matrix multiplication optimized for small matrices.

    Computes C[i] = A[i] @ B[i] for each batch index i.

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)
        out_dtype: Optional output dtype

    Returns:
        Output tensor of shape (batch, M, N)
    """
    assert a.dim() == 3 and b.dim() == 3, "Inputs must be 3D (batch, M, K) and (batch, K, N)"
    assert a.shape[0] == b.shape[0], f"Batch size mismatch: {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[2] == b.shape[1], f"K dimension mismatch: {a.shape[2]} vs {b.shape[1]}"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"

    batch, M, K = a.shape
    _, _, N = b.shape

    out_dtype = out_dtype or a.dtype
    c = torch.empty((batch, M, N), dtype=out_dtype, device=a.device)

    # Grid: one program per (batch, m_tile, n_tile)
    def grid(meta):
        return (
            batch,
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    _batch_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )

    return c


@triton.jit
def _batch_matmul_bwd_a_kernel(
    GRAD_OUT,  # (batch, M, N)
    B,         # (batch, K, N)
    GRAD_A,    # (batch, M, K)
    M, N, K,
    stride_gob, stride_gom, stride_gon,
    stride_bb, stride_bk, stride_bn,
    stride_gab, stride_gam, stride_gak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward kernel for A: grad_A = grad_out @ B^T"""
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    go_batch_ptr = GRAD_OUT + pid_batch * stride_gob
    b_batch_ptr = B + pid_batch * stride_bb
    ga_batch_ptr = GRAD_A + pid_batch * stride_gab

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n

        # Load grad_out tile: (BLOCK_M, BLOCK_N)
        go_ptrs = go_batch_ptr + offs_m[:, None] * stride_gom + n_offs[None, :] * stride_gon
        go_mask = (offs_m[:, None] < M) & (n_offs[None, :] < N)
        go = tl.load(go_ptrs, mask=go_mask, other=0.0)

        # Load B^T tile: (BLOCK_N, BLOCK_K) - transpose by swapping indices
        b_ptrs = b_batch_ptr + n_offs[:, None] * stride_bn + offs_k[None, :] * stride_bk
        b_mask = (n_offs[:, None] < N) & (offs_k[None, :] < K)
        b_t = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(go, b_t, acc)

    ga_ptrs = ga_batch_ptr + offs_m[:, None] * stride_gam + offs_k[None, :] * stride_gak
    ga_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(ga_ptrs, acc.to(GRAD_A.dtype.element_ty), mask=ga_mask)


@triton.jit
def _batch_matmul_bwd_b_kernel(
    GRAD_OUT,  # (batch, M, N)
    A,         # (batch, M, K)
    GRAD_B,    # (batch, K, N)
    M, N, K,
    stride_gob, stride_gom, stride_gon,
    stride_ab, stride_am, stride_ak,
    stride_gbb, stride_gbk, stride_gbn,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for B: grad_B = A^T @ grad_out"""
    pid_batch = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    go_batch_ptr = GRAD_OUT + pid_batch * stride_gob
    a_batch_ptr = A + pid_batch * stride_ab
    gb_batch_ptr = GRAD_B + pid_batch * stride_gbb

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + offs_m

        # Load A^T tile: (BLOCK_K, BLOCK_M) - transpose by swapping indices
        a_ptrs = a_batch_ptr + m_offs[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & (offs_k[None, :] < K)
        a_t = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a_t = tl.trans(a_t)

        # Load grad_out tile: (BLOCK_M, BLOCK_N)
        go_ptrs = go_batch_ptr + m_offs[:, None] * stride_gom + offs_n[None, :] * stride_gon
        go_mask = (m_offs[:, None] < M) & (offs_n[None, :] < N)
        go = tl.load(go_ptrs, mask=go_mask, other=0.0)

        acc = tl.dot(a_t, go, acc)

    gb_ptrs = gb_batch_ptr + offs_k[:, None] * stride_gbk + offs_n[None, :] * stride_gbn
    gb_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(gb_ptrs, acc.to(GRAD_B.dtype.element_ty), mask=gb_mask)


def batch_matmul_backward(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for batched matmul.

    Args:
        grad_out: Gradient of output (batch, M, N)
        a: Original A tensor (batch, M, K)
        b: Original B tensor (batch, K, N)

    Returns:
        (grad_a, grad_b)
    """
    # grad_A = grad_out @ B^T
    grad_a = torch.bmm(grad_out, b.transpose(1, 2))
    # grad_B = A^T @ grad_out
    grad_b = torch.bmm(a.transpose(1, 2), grad_out)

    return grad_a, grad_b


class BatchMatmulFunc(torch.autograd.Function):
    """Autograd function for batched matmul."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return batch_matmul(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a, grad_b = batch_matmul_backward(grad_out, a, b)
        return grad_a, grad_b


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiplication with autograd support.

    Optimized for small matrices common in robotics applications.

    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)

    Returns:
        Output tensor of shape (batch, M, N)
    """
    return BatchMatmulFunc.apply(a, b)
