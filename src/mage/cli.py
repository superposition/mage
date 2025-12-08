"""CLI entry point for mage."""

import argparse

import torch

from mage import add, fma, matmul, relu, softmax
from mage.bench import bench, bench_all, bench_backward, bench_matmul_sweep, print_results


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

    # Matmul
    a = torch.randn(256, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 256, device=device, dtype=torch.float16)
    out = matmul(a, b)
    diff = (out - torch.matmul(a, b)).abs().max().item()
    print(f"matmul:  max diff = {diff:.2e} {'OK' if diff < 1e-2 else 'FAIL'}")


def benchmark(op: str = "all"):
    """Run benchmarks comparing Triton vs PyTorch.

    Args:
        op: Operation to benchmark ('all', 'add', 'matmul', 'relu', 'softmax', 'backward')
    """
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    if op == "all":
        all_results = bench_all(device=str(device))
        for name, results in all_results.items():
            print_results(results, title=f"{name.upper()} Benchmarks")
        return

    if op == "matmul":
        print("Matmul Benchmarks (fp16)")
        print("=" * 80)
        results = bench_matmul_sweep(device=str(device))
        print_results(results)
        return

    if op == "add":
        print("Add Benchmarks")
        print("=" * 80)
        sizes = [2**i for i in range(12, 25)]  # 4K to 16M elements
        results = []
        for size in sizes:
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            results.append(bench("add", add, lambda a, b: a + b, x, y))
        print_results(results)
        return

    if op == "relu":
        print("ReLU Benchmarks")
        print("=" * 80)
        sizes = [2**i for i in range(12, 25)]
        results = []
        for size in sizes:
            x = torch.randn(size, device=device)
            results.append(bench("relu", relu, torch.relu, x))
        print_results(results)
        return

    if op == "softmax":
        print("Softmax Benchmarks")
        print("=" * 80)
        results = []
        for rows in [32, 64, 128, 256, 512, 1024, 2048]:
            cols = 1024
            x = torch.randn(rows, cols, device=device)
            results.append(
                bench(f"softmax-{rows}x{cols}", softmax, lambda x: torch.softmax(x, dim=1), x)
            )
        print_results(results)
        return

    if op == "backward":
        print("Backward Pass Benchmarks")
        print("=" * 80)
        results = []

        # Add backward
        x = torch.randn(1_000_000, device=device)
        y = torch.randn(1_000_000, device=device)
        results.append(bench_backward("add", add, lambda a, b: a + b, x, y))

        # ReLU backward
        x = torch.randn(1_000_000, device=device)
        results.append(bench_backward("relu", relu, torch.relu, x))

        # FMA backward
        a = torch.randn(1_000_000, device=device)
        x = torch.randn(1_000_000, device=device)
        y = torch.randn(1_000_000, device=device)
        results.append(bench_backward("fma", fma, lambda a, x, y: a * x + y, a, x, y))

        # Matmul backward
        a = torch.randn(512, 512, device=device, dtype=torch.float16)
        b = torch.randn(512, 512, device=device, dtype=torch.float16)
        results.append(bench_backward("matmul-512", matmul, torch.matmul, a, b))

        a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        results.append(bench_backward("matmul-1024", matmul, torch.matmul, a, b))

        print_results(results)
        return

    print(f"Unknown operation: {op}")
    print("Available: all, add, matmul, relu, softmax, backward")


def main():
    parser = argparse.ArgumentParser(
        description="Mage - High-performance Triton CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mage demo              Run quick correctness demo
  mage bench             Run all benchmarks
  mage bench matmul      Run matmul benchmarks only
  mage bench backward    Run backward pass benchmarks
        """,
    )
    parser.add_argument(
        "command", nargs="?", default="demo", choices=["demo", "bench"], help="Command to run"
    )
    parser.add_argument(
        "operation",
        nargs="?",
        default="all",
        choices=["all", "add", "matmul", "relu", "softmax", "backward"],
        help="Operation to benchmark (only for bench command)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    if args.command == "demo":
        demo()
    elif args.command == "bench":
        benchmark(args.operation)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
