"""Repeat the PyTorch/Triton/Rust comparison with rotating measurement order.

This measures warmed, resident-input forward operations, not service latency.
Each observation retains all event samples and a full-element correctness check.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
import triton

from experiment import DEFAULTS, generate, measure, precision, reference, resolve_implementation, source_hashes
from triton_target import implementation


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu",
        "--format=csv"], text=True).strip()


def run(args):
    precision()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = Path(__file__).resolve().parent
    entry = resolve_implementation(args.implementation)
    binary = (args.binary or entry["binary"]).resolve()
    data = {
        "experiment_id": f"{args.experiment}-comparison", "utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(), "platform": platform.platform(),
        "precision": "FP32; PyTorch TF32 disabled; Triton dot input_precision=ieee",
        "timing": "CUDA-event span per operation; resident inputs; includes possible host submission gaps",
        "excluded": ["compilation", "input transfers", "Rust process startup", "end-to-end service work"],
        "rounds": args.rounds, "iterations_per_round": args.iterations, "warmup_per_round": args.warmup,
        "implementation": args.implementation, "upstream": entry["upstream"],
        "source_sha256": {**{name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
                             for name in ("comparison.py", "triton_target.py", "experiment.py")},
                          **source_hashes(entry["source"])},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "gpu_state_before": gpu_state(), "results": [],
    }
    selected = {op: dims for op, dims in DEFAULTS.items() if not args.ops or op in args.ops}
    for op, dims in selected.items():
        directory = args.output / op
        generate(directory, op, dims, warmup=args.warmup, iterations=args.iterations)
        _, python_fn = reference(directory)
        _, triton_fn = implementation(directory)
        expected = python_fn().cpu().numpy()
        row = {"op": op, "dims": dims,
               "input_hashes": json.loads((directory / "inputs.sha256.json").read_text()), "rounds": []}
        for round_index in range(args.rounds):
            order = ["python", "triton", "rust"]
            shift = round_index % len(order)
            order = order[shift:] + order[:shift]
            observation = {"index": round_index, "order": order, "gpu_state": gpu_state(), "implementations": {}}
            for language in order:
                torch.cuda.synchronize()
                if language == "rust":
                    result = subprocess.run([str(binary), str(directory)], text=True, capture_output=True)
                    (directory / f"rust-round-{round_index}.log").write_text(result.stdout + result.stderr)
                    result.check_returncode()
                    actual = np.fromfile(directory / "rust-output.bin", dtype="<f4").reshape(expected.shape)
                    timing = json.loads((directory / "rust-timing.json").read_text())
                else:
                    fn = python_fn if language == "python" else triton_fn
                    timing = measure(fn, args.warmup, args.iterations)
                    actual = fn().cpu().numpy()
                passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=1e-4, atol=1e-4))
                checked = {"passed": passed, "max_abs_error": float(np.abs(actual - expected).max()),
                           "rtol": 1e-4, "atol": 1e-4, "checked_elements": int(expected.size)}
                if not passed:
                    raise AssertionError((op, language, checked))
                observation["implementations"][language] = {**timing, "correctness": checked}
                print(f"{op} round={round_index + 1} {language}: {timing['event_mean_us']:.3f} us; correctness passed", flush=True)
            row["rounds"].append(observation)
        data["results"].append(row)
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    data["gpu_state_after"] = gpu_state()
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    from experiment import IMPLEMENTATIONS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="oxide",
                        help="which native implementation to compare against PyTorch and Triton")
    parser.add_argument("--binary", type=Path, help="override the implementation's binary path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="mage-001", help="result namespace in results.json")
    parser.add_argument("--ops", nargs="+", choices=sorted(DEFAULTS), help="restrict the run to these operations")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()
    if args.rounds < 1 or not 1 <= args.iterations <= 100000 or not 0 <= args.warmup <= 10000:
        parser.error("positive rounds, 1..100000 iterations, and 0..10000 warmup iterations required")
    args.output = args.output.resolve()
    run(args)
