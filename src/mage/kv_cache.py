"""KV Cache operations for efficient autoregressive decoding.

The KV cache stores previously computed keys and values to avoid
redundant computation during inference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_cache_update_kernel(
    CACHE,  # Existing cache (batch*heads, max_seq, head_dim) - flattened
    NEW,    # New K or V (batch*heads, new_seq, head_dim) - flattened
    seq_pos,  # Starting position in cache
    head_dim,
    stride_c_bh, stride_c_s, stride_c_d,  # Cache strides: batch*head, seq, dim
    stride_n_bh, stride_n_s, stride_n_d,  # New strides: batch*head, seq, dim
    BLOCK_SIZE: tl.constexpr,
):
    """Update KV cache with new values at specified position."""
    # Program IDs: (batch*head, new_seq_idx)
    pid_bh = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # Compute offsets for flattened 3D tensors
    cache_seq_idx = seq_pos + pid_seq
    new_offset = pid_bh * stride_n_bh + pid_seq * stride_n_s
    cache_offset = pid_bh * stride_c_bh + cache_seq_idx * stride_c_s

    # Copy all dimensions
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim

    # Load from new tensor and store to cache
    vals = tl.load(NEW + new_offset + offs * stride_n_d, mask=mask, other=0.0)
    tl.store(CACHE + cache_offset + offs * stride_c_d, vals, mask=mask)


class KVCache:
    """Key-Value cache for efficient autoregressive decoding.

    Stores past keys and values to avoid recomputation during inference.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        """Initialize KV cache.

        Args:
            batch_size: Batch size
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            head_dim: Dimension per head
            dtype: Data type for cache
            device: Device for cache tensors
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Allocate cache tensors
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        # Track current position
        self.seq_pos = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new keys and values.

        Args:
            k: New keys of shape (batch, num_heads, new_seq, head_dim)
            v: New values of shape (batch, num_heads, new_seq, head_dim)

        Returns:
            Tuple of (cached_k, cached_v) containing all keys/values up to current position
        """
        batch, num_heads, new_seq, head_dim = k.shape

        assert batch == self.batch_size, f"Batch size mismatch: {batch} vs {self.batch_size}"
        assert num_heads == self.num_heads, f"Num heads mismatch: {num_heads} vs {self.num_heads}"
        assert head_dim == self.head_dim, f"Head dim mismatch: {head_dim} vs {self.head_dim}"
        assert self.seq_pos + new_seq <= self.max_seq_len, \
            f"Cache overflow: {self.seq_pos} + {new_seq} > {self.max_seq_len}"

        # Flatten batch and head dimensions
        k_flat = k.view(batch * num_heads, new_seq, head_dim)
        v_flat = v.view(batch * num_heads, new_seq, head_dim)
        k_cache_flat = self.k_cache.view(batch * num_heads, self.max_seq_len, head_dim)
        v_cache_flat = self.v_cache.view(batch * num_heads, self.max_seq_len, head_dim)

        # Block size for kernel
        BLOCK_SIZE = triton.next_power_of_2(head_dim)

        # Grid: one program per (batch*head, new_seq_pos)
        grid = (batch * num_heads, new_seq)

        # Update K cache
        _kv_cache_update_kernel[grid](
            k_cache_flat, k_flat,
            self.seq_pos, head_dim,
            k_cache_flat.stride(0), k_cache_flat.stride(1), k_cache_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            BLOCK_SIZE,
        )

        # Update V cache
        _kv_cache_update_kernel[grid](
            v_cache_flat, v_flat,
            self.seq_pos, head_dim,
            v_cache_flat.stride(0), v_cache_flat.stride(1), v_cache_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            BLOCK_SIZE,
        )

        # Update position
        self.seq_pos += new_seq

        # Return cached K, V up to current position
        return (
            self.k_cache[:, :, :self.seq_pos, :],
            self.v_cache[:, :, :self.seq_pos, :],
        )

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current cached keys and values.

        Returns:
            Tuple of (k_cache, v_cache) up to current sequence position
        """
        return (
            self.k_cache[:, :, :self.seq_pos, :],
            self.v_cache[:, :, :self.seq_pos, :],
        )

    def reset(self):
        """Reset cache to empty state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_pos = 0

    @property
    def current_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self.seq_pos


def create_kv_cache(
    batch_size: int,
    num_heads: int,
    max_seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> KVCache:
    """Create a new KV cache.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head
        dtype: Data type for cache
        device: Device for cache tensors

    Returns:
        KVCache instance
    """
    return KVCache(batch_size, num_heads, max_seq_len, head_dim, dtype, device)
