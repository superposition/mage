"""CLI entry point for mage."""

import argparse
import sys

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


def profile(
    script: str,
    args: list[str] | None = None,
    backend: str = "triton",
    columns: list[str] | None = None,
    group_by: str | None = None,
    no_persist: bool = False,
    db_path: str | None = None,
    analyze: bool = False,
) -> int:
    """Profile a Python script using nsys or ncu.

    Args:
        script: Path to the Python script to profile
        args: Additional arguments to pass to the script
        backend: Profiler backend ('nsys' or 'ncu')
        columns: List of columns to display
        group_by: Field to group metrics by
        no_persist: Don't save to database
        db_path: Custom database path
        analyze: Run memory analysis after profiling

    Returns:
        Exit code (0 for success)
    """
    from mage.profiler import get_backend, ProfilerTUI, ProfileDB, print_memory_report

    # Get the profiler backend
    try:
        profiler = get_backend(backend)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not profiler.is_available():
        print(f"Error: {backend} is not available on this system")
        print(f"Make sure NVIDIA {backend} is installed and in your PATH")
        return 1

    # Set up TUI
    tui = ProfilerTUI(columns=columns, group_by=group_by)
    tui.start_session(f"python {script}", backend=backend)

    print(f"Profiling {script} with {backend}...")
    print("This may take a moment...\n")

    # Run profiler
    try:
        for metric in profiler.run(script, args, callback=tui.add_metric):
            pass  # Metrics added via callback
    except KeyboardInterrupt:
        print("\nProfiling interrupted")
    except Exception as e:
        print(f"Error during profiling: {e}")
        return 1
    finally:
        tui.finish()
        if hasattr(profiler, "cleanup"):
            profiler.cleanup()

    # Display results
    tui.print_final()
    tui.print_summary()

    # Memory analysis
    if analyze and tui.aggregator.metrics:
        print("\n")
        print_memory_report(tui.aggregator.metrics)

    # Save to database
    if not no_persist and tui.session:
        try:
            db = ProfileDB(db_path) if db_path else ProfileDB()
            session_id = db.save_session(tui.session)
            print(f"\nSession saved to database (id={session_id})")
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Mage - High-performance Triton CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick correctness demo")

    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument(
        "operation",
        nargs="?",
        default="all",
        choices=["all", "add", "matmul", "relu", "softmax", "backward"],
        help="Operation to benchmark",
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a Python script with nsys or ncu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mage profile script.py
  mage profile --backend ncu script.py
  mage profile --columns kernel,duration,occupancy script.py
  mage profile --group-by kernel_name script.py
        """,
    )
    profile_parser.add_argument("script", help="Python script to profile")
    profile_parser.add_argument(
        "script_args", nargs="*", help="Arguments to pass to the script"
    )
    profile_parser.add_argument(
        "--backend", "-b",
        choices=["triton", "nsys", "ncu"],
        default="triton",
        help="Profiler backend (default: triton, no special permissions needed)",
    )
    profile_parser.add_argument(
        "--columns", "-c",
        help="Comma-separated list of columns to display",
    )
    profile_parser.add_argument(
        "--group-by", "-g",
        help="Group metrics by field (e.g., kernel_name)",
    )
    profile_parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't save results to database",
    )
    profile_parser.add_argument(
        "--db",
        help="Custom database path",
    )
    profile_parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Run memory analysis after profiling",
    )

    # Stress test command
    stress_parser = subparsers.add_parser(
        "stress",
        help="Run GPU stress tests (memory, compute, cache, roofline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mage stress              # Full stress test suite
  mage stress --quick      # Quick version (~1 minute)
        """,
    )
    stress_parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick version with fewer iterations",
    )

    args = parser.parse_args()

    # Default to demo if no command given
    if args.command is None:
        args.command = "demo"

    # Profile command doesn't require CUDA check (profiler handles it)
    if args.command == "profile":
        columns = args.columns.split(",") if args.columns else None
        return profile(
            script=args.script,
            args=args.script_args or None,
            backend=args.backend,
            columns=columns,
            group_by=args.group_by,
            no_persist=args.no_persist,
            db_path=args.db,
            analyze=args.analyze,
        )

    # Other commands require CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    if args.command == "demo":
        demo()
    elif args.command == "bench":
        benchmark(args.operation)
    elif args.command == "stress":
        from mage.stress import run_stress_suite
        run_stress_suite(quick=args.quick)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
