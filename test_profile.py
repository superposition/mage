"""Simple script to profile Triton kernels."""
import torch
from mage import add, relu, softmax, fma

def main():
    device = torch.device("cuda")

    # Warmup
    x = torch.randn(1000, device=device)
    y = torch.randn(1000, device=device)
    _ = add(x, y)
    torch.cuda.synchronize()

    # Profile vector add at different sizes
    for size in [100_000, 1_000_000, 10_000_000]:
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)

        for _ in range(10):
            out = add(x, y)
        torch.cuda.synchronize()

    # Profile ReLU
    for size in [100_000, 1_000_000, 10_000_000]:
        x = torch.randn(size, device=device)
        for _ in range(10):
            out = relu(x)
        torch.cuda.synchronize()

    # Profile FMA (fused multiply-add)
    for size in [100_000, 1_000_000]:
        a = torch.randn(size, device=device)
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        for _ in range(10):
            out = fma(a, x, y)
        torch.cuda.synchronize()

    # Profile Softmax
    for rows in [256, 1024, 4096]:
        x = torch.randn(rows, 1024, device=device)
        for _ in range(10):
            out = softmax(x)
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
