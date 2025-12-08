"""Tests for benchmarking utilities."""

import pytest
import torch

from mage import add, matmul, relu
from mage.bench import BenchResult, bench, bench_backward, bench_matmul_sweep


class TestBenchResult:
    def test_speedup_calculation(self):
        result = BenchResult("test", 1000, triton_ms=1.0, torch_ms=2.0)
        assert result.speedup == 2.0

    def test_str_format(self):
        result = BenchResult("test", 1000, triton_ms=1.234, torch_ms=2.345)
        output = str(result)
        assert "test" in output
        assert "1.234" in output
        assert "2.345" in output


class TestBench:
    def test_basic_bench(self, device):
        x = torch.randn(10000, device=device)
        y = torch.randn(10000, device=device)

        result = bench("add", add, lambda a, b: a + b, x, y, warmup=2, rep=5)

        assert result.name == "add"
        assert result.size == 10000
        assert result.triton_ms > 0
        assert result.torch_ms > 0


class TestBenchBackward:
    def test_backward_bench(self, device):
        x = torch.randn(10000, device=device)
        y = torch.randn(10000, device=device)

        result = bench_backward("add", add, lambda a, b: a + b, x, y, warmup=2, rep=5)

        assert "fwd+bwd" in result.name
        assert result.triton_ms > 0
        assert result.torch_ms > 0


class TestBenchMatmulSweep:
    def test_matmul_sweep_runs(self, device):
        # Use small sizes for testing
        results = bench_matmul_sweep(sizes=[32, 64], device=str(device))

        assert len(results) == 2
        assert all(isinstance(r, BenchResult) for r in results)
        assert "matmul-32" in results[0].name
        assert "matmul-64" in results[1].name
