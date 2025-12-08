"""CLI entry point for mage."""

import argparse

import torch

from mage import add, fma, relu, softmax
from mage.bench import bench, print_results


def demo():
    """Run a quick demo of all kernels."""
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Vector add
    x = torch.randn(10000, device=device)
    y = torch.randn(10000, device=device)
    out = add(x, y)
    diff = (out - (x + y)).abs().max().item()
    print(f"add:     max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'FAIL'}")

    # FMA
    a = torch.randn(10000, device=device)
    out = fma(a, x, y)
    diff = (out - (a * x + y)).abs().max().item()
    print(f"fma:     max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'FAIL'}")

    # ReLU
    out = relu(x)
    diff = (out - torch.relu(x)).abs().max().item()
    print(f"relu:    max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'FAIL'}")

    # Softmax
    m = torch.randn(128, 256, device=device)
    out = softmax(m)
    diff = (out - torch.softmax(m, dim=1)).abs().max().item()
    print(f"softmax: max diff = {diff:.2e} {'OK' if diff < 1e-5 else 'FAIL'}")


def benchmark():
    """Run benchmarks comparing Triton vs PyTorch."""
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    sizes = [2**i for i in range(12, 25)]  # 4K to 16M elements
    results = []

    for size in sizes:
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        results.append(bench("add", add, lambda a, b: a + b, x, y))

    print_results(results)


def main():
    parser = argparse.ArgumentParser(description="Mage - Triton CUDA kernels")
    parser.add_argument("command", nargs="?", default="demo", choices=["demo", "bench"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    if args.command == "demo":
        demo()
    elif args.command == "bench":
        benchmark()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
