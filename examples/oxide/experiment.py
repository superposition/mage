"""Generate identical FP32 inputs, validate Rust, and time the PyTorch reference.

Run with the Mage environment. Nsight capture uses python_target.py or the native
binary with --capture; plain CUDA-event samples must be collected separately.
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
import torch.nn.functional as F

DEFAULTS = {"matmul": [1024, 1024, 1024], "gelu": [4096, 768],
            "layernorm": [4096, 768], "triangle": [128, 32], "neighbor": [4096, 64, 65536]}
SMALL = {"matmul": [17, 19, 23], "gelu": [3, 257], "layernorm": [3, 257],
         "triangle": [7, 5], "neighbor": [7, 9, 23]}

EXAMPLES = Path(__file__).resolve().parent.parent

# Native implementations this harness drives. They share the input files, the
# binary CLI and the result layout, so a run names the implementation it used
# rather than assuming one.
IMPLEMENTATIONS = {
    "oxide": {
        "binary": EXAMPLES / "oxide/target/release/mage-oxide",
        "source": EXAMPLES / "oxide",
        "upstream": "cuda-oxide @26754ae52c26c097dc1c465a1e42c4c5d05a3d40",
    },
    "cutile": {
        "binary": EXAMPLES / "cutile/target/release/mage-cutile",
        "source": EXAMPLES / "cutile",
        "upstream": "cutile 0.3.1 (crates.io)",
    },
}


def resolve_implementation(name):
    if name not in IMPLEMENTATIONS:
        raise SystemExit(f"unknown implementation {name!r}; expected one of {', '.join(IMPLEMENTATIONS)}")
    return IMPLEMENTATIONS[name]


def toolchain(source):
    """The pinned toolchain channel a native implementation builds with."""
    for line in (source / "rust-toolchain.toml").read_text().splitlines():
        if line.startswith("channel"):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def source_hashes(source):
    files = {"src/main.rs": source / "src/main.rs",
             "Cargo.lock": source / "Cargo.lock",
             "experiment.py": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def generate(directory, op, dims, *, seed=20260910, warmup=10, iterations=100, constant=False):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    def write(name, shape):
        data = np.full(shape, 0.25, dtype="<f4") if constant else rng.normal(0, .25, shape).astype("<f4")
        data.tofile(directory / name)
    if op == "matmul":
        m, n, k = dims
        write("a.bin", (m, k)); write("b.bin", (k, n))
    elif op in {"gelu", "layernorm"}:
        rows, width = dims
        write("a.bin", (rows, width)); write("b.bin", (width,))
        if op == "layernorm":
            write("c.bin", (width,))
    elif op == "triangle":
        n, channels = dims
        write("a.bin", (n, n, channels)); write("b.bin", (n, n, channels))
    elif op == "neighbor":
        n, width, edges = dims
        write("a.bin", (n, width)); write("b.bin", (edges,))
        destinations = np.sort(rng.integers(0, n, size=edges))
        ptr = np.concatenate(([0], np.cumsum(np.bincount(destinations, minlength=n)))).astype("<u4")
        ptr.tofile(directory / "rowptr.bin")
        rng.integers(0, n, size=edges, dtype=np.uint32).astype("<u4").tofile(directory / "indices.bin")
    manifest = {"op": op, "dims": dims, "warmup": warmup, "iterations": iterations}
    (directory / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.glob("*.bin"))
              if f.name in {"a.bin", "b.bin", "c.bin", "rowptr.bin", "indices.bin"}}
    (directory / "inputs.sha256.json").write_text(json.dumps({"seed": seed, "sha256": hashes}, indent=2) + "\n")


def reference(directory):
    precision()
    manifest = json.loads((directory / "input.json").read_text())
    op, dims = manifest["op"], manifest["dims"]
    def read(name, shape):
        return torch.from_numpy(np.fromfile(directory / name, dtype="<f4").reshape(shape)).cuda()
    if op == "matmul":
        m, n, k = dims
        a, b = read("a.bin", (m, k)), read("b.bin", (k, n))
        fn = lambda: a @ b
    elif op in {"gelu", "layernorm"}:
        rows, width = dims
        a, b = read("a.bin", (rows, width)), read("b.bin", (width,))
        if op == "gelu":
            fn = lambda: F.gelu(a + b, approximate="tanh")
        else:
            c = read("c.bin", (width,))
            fn = lambda: F.layer_norm(a, (width,), b, c, eps=1e-5)
    elif op == "triangle":
        n, channels = dims
        a, b = read("a.bin", (n, n, channels)), read("b.bin", (n, n, channels))
        fn = lambda: torch.einsum("ikc,jkc->ijc", a, b)
    else:
        n, width, edges = dims
        a, weights = read("a.bin", (n, width)), read("b.bin", (edges,))
        ptr = np.fromfile(directory / "rowptr.bin", dtype="<u4").astype(np.int64)
        indices = torch.from_numpy(np.fromfile(directory / "indices.bin", dtype="<u4").astype(np.int64)).cuda()
        dest = torch.from_numpy(np.repeat(np.arange(n), np.diff(ptr))).cuda()
        fn = lambda: torch.zeros_like(a).index_add_(0, dest, weights[:, None] * a[indices])
    return manifest, fn


def measure(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record(); fn(); end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {"samples_us": samples, "event_mean_us": float(np.mean(samples)),
            "event_median_us": float(np.median(samples)), "warmup": warmup, "iterations": iterations}


def validate(directory):
    manifest, fn = reference(directory)
    expected = fn().cpu().numpy()
    actual = np.fromfile(directory / "rust-output.bin", dtype="<f4").reshape(expected.shape)
    difference = np.abs(actual - expected)
    passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=1e-4, atol=1e-4))
    result = {"op": manifest["op"], "dims": manifest["dims"], "passed": passed,
              "rtol": 1e-4, "atol": 1e-4, "max_abs_error": float(difference.max()),
              "reference": "PyTorch FP32, TF32 disabled", "checked_elements": int(expected.size)}
    (directory / "correctness.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed: raise AssertionError(result)
    return result


def run_suite(binary, output, small=False, implementation="oxide", experiment_id="mage-001"):
    precision()
    cases = [(op, dims, False, op) for op, dims in (SMALL if small else DEFAULTS).items()]
    if small:
        cases += [("layernorm", [2, 1], True, "layernorm-constant-one"),
                  ("layernorm", [4, 768], True, "layernorm-constant"),
                  ("neighbor", [7, 9, 0], False, "neighbor-no-edges"),
                  ("neighbor", [1, 5, 9], False, "neighbor-duplicates"),
                  ("matmul", [1, 1, 1], False, "matmul-one")]
    results = []
    for op, dims, constant, name in cases:
        directory = output / name
        generate(directory, op, dims, iterations=5 if small else 100, constant=constant)
        process = subprocess.run([str(binary), str(directory)], text=True, capture_output=True)
        (directory / "rust.log").write_text(process.stdout + process.stderr)
        process.check_returncode()
        correctness = validate(directory)
        manifest, fn = reference(directory)
        pytorch = measure(fn, manifest["warmup"], manifest["iterations"])
        (directory / "python-timing.json").write_text(json.dumps(pytorch, indent=2) + "\n")
        native = json.loads((directory / "rust-timing.json").read_text())
        results.append({**correctness, "case": name, "rust_event_mean_us": native["event_mean_us"],
                        "python_event_mean_us": pytorch["event_mean_us"],
                        "rust_samples_us": native["samples_us"], "python_samples_us": pytorch["samples_us"],
                        "rust_run_id": native["run_id"],
                        "input_hashes": json.loads((directory / "inputs.sha256.json").read_text())})
        print(json.dumps({k: v for k, v in results[-1].items() if k not in
                          {"rust_samples_us", "python_samples_us", "input_hashes"}}), flush=True)
    entry = resolve_implementation(implementation)
    metadata = {"experiment_id": experiment_id, "utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "platform": platform.platform(),
        "implementation": implementation, "upstream": entry["upstream"],
        "rust_toolchain": toolchain(entry["source"]), "architecture": "sm_89",
        "source_sha256": source_hashes(entry["source"]),
        "gpu_state": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem", "--format=csv"], text=True).strip(),
        "timing": "CUDA events, inputs resident on device; host submission may affect short kernels",
        "results": results}
    if implementation == "oxide":
        # Published mage-001..003 records carry this field; keep it for them.
        metadata["oxide_revision"] = "26754ae52c26c097dc1c465a1e42c4c5d05a3d40"
    (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="oxide",
                        help="which native implementation to run")
    parser.add_argument("--binary", type=Path, help="override the implementation's binary path")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mage-001"))
    parser.add_argument("--experiment", default="mage-001", help="result namespace in results.json")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    binary = args.binary or resolve_implementation(args.implementation)["binary"]
    run_suite(binary.resolve(), args.output.resolve(), args.small, args.implementation, args.experiment)


if __name__ == "__main__":
    main()
