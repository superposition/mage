import pytest
import torch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.float32


@pytest.fixture
def rtol():
    return 1e-5


@pytest.fixture
def atol():
    return 1e-5
