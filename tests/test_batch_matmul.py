import pytest
import torch

from mage.batch_matmul import batch_matmul, bmm


class TestBatchMatmulForward:
    def test_basic(self, device):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32)

        result = batch_matmul(a, b)
        expected = torch.bmm(a, b)

        assert torch.allclose(result, expected, atol=5e-2, rtol=1e-2)

    def test_output_shape(self, device):
        batch, M, K, N = 100, 32, 16, 64
        a = torch.randn(batch, M, K, device=device)
        b = torch.randn(batch, K, N, device=device)

        result = batch_matmul(a, b)
        assert result.shape == (batch, M, N)

    def test_large_batch(self, device):
        """Test with large batch size typical in robotics."""
        batch, M, K, N = 1000, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32)

        result = batch_matmul(a, b)
        expected = torch.bmm(a, b)

        assert torch.allclose(result, expected, atol=5e-2, rtol=1e-2)

    def test_rectangular_matrices(self, device):
        batch = 64
        for M, K, N in [(8, 16, 32), (32, 8, 16), (16, 32, 8)]:
            a = torch.randn(batch, M, K, device=device, dtype=torch.float32)
            b = torch.randn(batch, K, N, device=device, dtype=torch.float32)

            result = batch_matmul(a, b)
            expected = torch.bmm(a, b)

            assert torch.allclose(result, expected, atol=5e-2), f"Failed for ({M}, {K}, {N})"

    def test_single_batch(self, device):
        a = torch.randn(1, 16, 16, device=device, dtype=torch.float32)
        b = torch.randn(1, 16, 16, device=device, dtype=torch.float32)

        result = batch_matmul(a, b)
        expected = torch.bmm(a, b)

        assert torch.allclose(result, expected, atol=5e-2)

    def test_small_matrices(self, device):
        """Test with very small matrices (robotics dynamics)."""
        batch = 500
        for size in [4, 6, 8, 12]:
            a = torch.randn(batch, size, size, device=device, dtype=torch.float32)
            b = torch.randn(batch, size, size, device=device, dtype=torch.float32)

            result = batch_matmul(a, b)
            expected = torch.bmm(a, b)

            assert torch.allclose(result, expected, atol=5e-2), f"Failed for size {size}"


class TestBatchMatmulBackward:
    def test_gradient_exists(self, device):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32, requires_grad=True)

        result = bmm(a, b)
        result.sum().backward()

        assert a.grad is not None
        assert b.grad is not None

    def test_gradient_shape(self, device):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32, requires_grad=True)

        result = bmm(a, b)
        result.sum().backward()

        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matches_torch_grad(self, device):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32, requires_grad=True)

        # Triton
        result = bmm(a, b)
        result.sum().backward()
        triton_grad_a = a.grad.clone()
        triton_grad_b = b.grad.clone()

        # Reset grads
        a.grad = None
        b.grad = None

        # PyTorch
        ref = torch.bmm(a, b)
        ref.sum().backward()

        assert torch.allclose(triton_grad_a, a.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(triton_grad_b, b.grad, atol=1e-3, rtol=1e-3)

    def test_backward_large_batch(self, device):
        """Test backward with large batch size."""
        batch, M, K, N = 500, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float32, requires_grad=True)

        # Triton
        result = bmm(a, b)
        result.sum().backward()
        triton_grad_a = a.grad.clone()
        triton_grad_b = b.grad.clone()

        # Reset and compare with PyTorch
        a.grad = None
        b.grad = None
        ref = torch.bmm(a, b)
        ref.sum().backward()

        assert torch.allclose(triton_grad_a, a.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(triton_grad_b, b.grad, atol=1e-3, rtol=1e-3)


class TestBatchMatmulDtypes:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_support(self, device, dtype):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=dtype)
        b = torch.randn(batch, K, N, device=device, dtype=dtype)

        result = batch_matmul(a, b)
        assert result.dtype == dtype

    def test_fp16_accuracy(self, device):
        batch, M, K, N = 32, 16, 16, 16
        a = torch.randn(batch, M, K, device=device, dtype=torch.float16)
        b = torch.randn(batch, K, N, device=device, dtype=torch.float16)

        result = batch_matmul(a, b)
        expected = torch.bmm(a.float(), b.float()).half()

        # fp16 has lower precision
        assert torch.allclose(result, expected, atol=5e-2, rtol=1e-2)


class TestBatchMatmulRobotics:
    """Tests specifically for robotics use cases."""

    def test_dynamics_batch(self, device):
        """Simulate batched robot dynamics: x_next = A @ x + B @ u"""
        num_robots = 1000
        state_dim = 12  # typical for rigid body
        control_dim = 6

        # State transition matrices per robot
        A = torch.randn(num_robots, state_dim, state_dim, device=device, dtype=torch.float32)
        x = torch.randn(num_robots, state_dim, 1, device=device, dtype=torch.float32)

        # Compute A @ x for all robots
        result = batch_matmul(A, x)
        expected = torch.bmm(A, x)

        assert result.shape == (num_robots, state_dim, 1)
        assert torch.allclose(result, expected, atol=5e-2)

    def test_jacobian_batch(self, device):
        """Simulate batched Jacobian computations."""
        num_configs = 500
        output_dim = 6  # end-effector pose
        input_dim = 7   # joint angles

        J = torch.randn(num_configs, output_dim, input_dim, device=device, dtype=torch.float32)
        dq = torch.randn(num_configs, input_dim, 1, device=device, dtype=torch.float32)

        # Compute J @ dq for all configurations
        result = batch_matmul(J, dq)
        expected = torch.bmm(J, dq)

        assert result.shape == (num_configs, output_dim, 1)
        assert torch.allclose(result, expected, atol=5e-2)

    def test_multi_agent_simulation(self, device):
        """Simulate multi-agent environment state updates."""
        num_agents = 256
        obs_dim = 32
        action_dim = 4
        hidden_dim = 64

        # Policy network layer: h = W @ obs
        W = torch.randn(num_agents, hidden_dim, obs_dim, device=device, dtype=torch.float32)
        obs = torch.randn(num_agents, obs_dim, 1, device=device, dtype=torch.float32)

        result = batch_matmul(W, obs)
        expected = torch.bmm(W, obs)

        assert result.shape == (num_agents, hidden_dim, 1)
        assert torch.allclose(result, expected, atol=5e-2)
