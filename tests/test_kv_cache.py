import pytest
import torch

from mage import KVCache, create_kv_cache


class TestKVCacheCreation:
    def test_create(self, device):
        cache = create_kv_cache(
            batch_size=2,
            num_heads=4,
            max_seq_len=128,
            head_dim=32,
            device=device,
        )
        assert cache.batch_size == 2
        assert cache.num_heads == 4
        assert cache.max_seq_len == 128
        assert cache.head_dim == 32
        assert cache.current_seq_len == 0

    def test_cache_shape(self, device):
        cache = create_kv_cache(2, 4, 128, 32, device=device)
        k, v = cache.get()
        assert k.shape == (2, 4, 0, 32)
        assert v.shape == (2, 4, 0, 32)

    def test_dtype(self, device):
        cache = create_kv_cache(2, 4, 128, 32, dtype=torch.float32, device=device)
        assert cache.k_cache.dtype == torch.float32
        assert cache.v_cache.dtype == torch.float32


class TestKVCacheUpdate:
    def test_single_update(self, device):
        cache = create_kv_cache(2, 4, 128, 32, device=device)

        k = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)

        k_out, v_out = cache.update(k, v)

        assert cache.current_seq_len == 10
        assert k_out.shape == (2, 4, 10, 32)
        assert v_out.shape == (2, 4, 10, 32)
        assert torch.allclose(k_out, k)
        assert torch.allclose(v_out, v)

    def test_multiple_updates(self, device):
        cache = create_kv_cache(2, 4, 128, 32, device=device)

        k1 = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        v1 = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        cache.update(k1, v1)

        k2 = torch.randn(2, 4, 5, 32, device=device, dtype=torch.float16)
        v2 = torch.randn(2, 4, 5, 32, device=device, dtype=torch.float16)
        k_out, v_out = cache.update(k2, v2)

        assert cache.current_seq_len == 15
        assert k_out.shape == (2, 4, 15, 32)
        assert v_out.shape == (2, 4, 15, 32)

        # Check that previous values are preserved
        assert torch.allclose(k_out[:, :, :10, :], k1)
        assert torch.allclose(v_out[:, :, :10, :], v1)
        # Check that new values are correct
        assert torch.allclose(k_out[:, :, 10:15, :], k2)
        assert torch.allclose(v_out[:, :, 10:15, :], v2)

    def test_autoregressive_simulation(self, device):
        """Simulate autoregressive decoding with single-token updates."""
        cache = create_kv_cache(1, 4, 64, 32, device=device)

        all_k = []
        all_v = []

        # Simulate 20 single-token decoding steps
        for _ in range(20):
            k = torch.randn(1, 4, 1, 32, device=device, dtype=torch.float16)
            v = torch.randn(1, 4, 1, 32, device=device, dtype=torch.float16)
            all_k.append(k)
            all_v.append(v)
            k_out, v_out = cache.update(k, v)

        assert cache.current_seq_len == 20
        assert k_out.shape == (1, 4, 20, 32)

        # Verify all cached values
        expected_k = torch.cat(all_k, dim=2)
        expected_v = torch.cat(all_v, dim=2)
        assert torch.allclose(k_out, expected_k)
        assert torch.allclose(v_out, expected_v)


class TestKVCacheReset:
    def test_reset(self, device):
        cache = create_kv_cache(2, 4, 128, 32, device=device)

        k = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        cache.update(k, v)

        assert cache.current_seq_len == 10

        cache.reset()

        assert cache.current_seq_len == 0
        k_out, v_out = cache.get()
        assert k_out.shape == (2, 4, 0, 32)


class TestKVCacheGet:
    def test_get_after_updates(self, device):
        cache = create_kv_cache(2, 4, 128, 32, device=device)

        k1 = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        v1 = torch.randn(2, 4, 10, 32, device=device, dtype=torch.float16)
        cache.update(k1, v1)

        k_out, v_out = cache.get()

        assert torch.allclose(k_out, k1)
        assert torch.allclose(v_out, v1)


class TestKVCacheDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        cache = create_kv_cache(2, 4, 64, 32, dtype=dtype, device=device)

        k = torch.randn(2, 4, 10, 32, device=device, dtype=dtype)
        v = torch.randn(2, 4, 10, 32, device=device, dtype=dtype)

        k_out, v_out = cache.update(k, v)

        assert k_out.dtype == dtype
        assert v_out.dtype == dtype
        assert torch.allclose(k_out, k)


class TestKVCacheEdgeCases:
    def test_full_cache(self, device):
        cache = create_kv_cache(1, 1, 10, 16, device=device)

        # Fill cache completely
        k = torch.randn(1, 1, 10, 16, device=device, dtype=torch.float16)
        v = torch.randn(1, 1, 10, 16, device=device, dtype=torch.float16)
        cache.update(k, v)

        assert cache.current_seq_len == 10

    def test_overflow_raises(self, device):
        cache = create_kv_cache(1, 1, 10, 16, device=device)

        k = torch.randn(1, 1, 10, 16, device=device, dtype=torch.float16)
        v = torch.randn(1, 1, 10, 16, device=device, dtype=torch.float16)
        cache.update(k, v)

        # Try to add more - should raise
        k2 = torch.randn(1, 1, 1, 16, device=device, dtype=torch.float16)
        v2 = torch.randn(1, 1, 1, 16, device=device, dtype=torch.float16)

        with pytest.raises(AssertionError, match="Cache overflow"):
            cache.update(k2, v2)
