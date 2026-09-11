"""Does an event-span win show up in GPU kernel time?

Measures one configuration against the committed kernels with Nsight Systems, using
the same instrument and bracket as `examples/oxide/profile_suite.py`: `NsysBackend`
with `capture_range="cuda"` around the binary's own `--capture` region. The published
mage-001..003 records are kernel time, so a change the loop accepted on event span has
to be re-measured here before it can be compared with those numbers.

    .venv/bin/python scripts/evolve_capture.py \
        --params '{"block": [64, 64], "k_step": 32, "quad_stage": true,
                   "staging": "guarded", "thread_tile": [4, 4], "transpose_a": true}' \
        --rounds 3 --iterations 100 --out artifacts/mage-005-nsys

Prints kernel microseconds per iteration for both arms and writes `capture.json` with
every capture, the kernel names, the input hashes, and the rendered source hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = REPO_ROOT / "examples" / "oxide" / "src" / "candidates.rs"
BINARY = REPO_ROOT / "examples" / "oxide" / "target" / "release" / "mage-oxide"
BUILD_COMMAND = ("source scripts/oxide-env.sh && cd examples/oxide && "
                 "CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89")
DEFAULT_SHAPE = (1024, 1024, 1024)


def load(name: str, base: Path):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = load("evolve_render", REPO_ROOT / "scripts")
experiment = load("experiment", REPO_ROOT / "examples" / "oxide")
sys.path.insert(0, str(REPO_ROOT / "src"))
from mage.profiler.backends import NsysBackend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", required=True,
                        help="matmul parameters to capture, as JSON (the retained config)")
    parser.add_argument("--op", default="matmul", choices=["matmul"])
    parser.add_argument("--shape", default=",".join(str(part) for part in DEFAULT_SHAPE))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", default="artifacts/mage-005-nsys")
    parser.add_argument("--triton", action="store_true",
                        help="also capture examples/oxide/triton_target.py on the same inputs")
    parser.add_argument("--pytorch", action="store_true",
                        help="also capture examples/oxide/python_target.py (cuBLAS) on the same inputs")
    parser.add_argument("--compare-params", default=None,
                        help="optional second generated configuration, captured as the {'new'} arm")
    args = parser.parse_args(argv)

    shape = [int(part) for part in args.shape.split(",")]
    out = (REPO_ROOT / args.out).resolve()
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    params = render.validate(args.op, json.loads(args.params), tuple(shape))
    compare = (render.validate(args.op, json.loads(args.compare_params), tuple(shape))
               if args.compare_params else None)
    other = render.DEFAULT_LAYERNORM if args.op == "matmul" else render.DEFAULT_MATMUL
    print("retained configuration:", render.params_json(params))
    if compare:
        print("compare configuration: ", render.params_json(compare))
    render.write_candidates(params, other, compare or params, other, generation=0, dest=CANDIDATES)
    build = subprocess.run(["bash", "-lc", BUILD_COMMAND], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit("build failed:\n" + (build.stdout + build.stderr)[-2000:])

    arms = {"committed": None, "retained": "best"}
    if compare:
        arms["compare"] = "new"
    if args.triton:
        arms["triton"] = None
    if args.pytorch:
        arms["pytorch"] = None
    directories = {}
    for arm, variant in arms.items():
        directory = work / f"{args.op}-{arm}"
        experiment.generate(directory, args.op, shape, seed=args.seed,
                            warmup=args.warmup, iterations=args.iterations)
        manifest = json.loads((directory / "input.json").read_text())
        if variant:
            manifest["variant"] = variant
        (directory / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
        directories[arm] = directory
    hashes = {arm: {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("a.bin", "b.bin")}
              for arm, directory in directories.items()}
    if len({json.dumps(value, sort_keys=True) for value in hashes.values()}) != 1:
        raise SystemExit("the arms did not receive identical inputs")

    rows = []
    for round_index in range(args.rounds):
        # Rotate the arms so each takes each position across the rounds; with two
        # arms this is the alternation the earlier captures used.
        names = list(arms)
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        for arm in order:
            scripts = {"triton": "triton_target.py", "pytorch": "python_target.py"}
            if arm in scripts:
                argv = [sys.executable,
                        str(REPO_ROOT / "examples" / "oxide" / scripts[arm]),
                        str(directories[arm]), "--iterations", str(args.iterations)]
            else:
                argv = [str(BINARY), str(directories[arm]),
                        "--iterations", str(args.iterations), "--capture"]
            backend = NsysBackend(capture_range="cuda",
                                  output_dir=out / f"r{round_index + 1}" / arm)
            try:
                metrics = list(backend.run_command(argv))
            finally:
                backend.cleanup()
            total_us = sum(metric.duration_us for metric in metrics)
            row = {"round": round_index + 1, "arm": arm, "launches": len(metrics),
                   "kernel_names": sorted({metric.kernel_name for metric in metrics}),
                   "total_us": total_us,
                   "per_iteration_us": total_us / args.iterations,
                   "report_dir": str(out / f"r{round_index + 1}" / arm)}
            rows.append(row)
            print("round %d %-9s launches=%3d kernel_time=%8.3f us/iter  %s" % (
                round_index + 1, arm, len(metrics), row["per_iteration_us"],
                ", ".join(row["kernel_names"])))
            if len(metrics) != args.iterations:
                print("warning: expected %d launches, captured %d" % (
                    args.iterations, len(metrics)), file=sys.stderr)

    medians = {}
    for arm in arms:
        values = sorted(row["per_iteration_us"] for row in rows if row["arm"] == arm)
        medians[arm] = values[len(values) // 2] if values else None
    summary = {
        "experiment_id": "kernel-time-capture",
        "instrument": "Nsight Systems, capture_range=cuda, binary --capture bracket",
        "op": args.op, "shape": shape, "iterations_per_capture": args.iterations,
        "rounds": args.rounds, "seed": args.seed,
        "arms": {"committed": "src/main.rs kernels",
                 "retained": "generated variant, params " + render.params_json(params),
                 **({"compare": "generated variant, params " + render.params_json(compare)}
                    if compare else {}),
                 **({"triton": "examples/oxide/triton_target.py, fixed tiles"}
                    if args.triton else {}),
                 **({"pytorch": "examples/oxide/python_target.py, library matmul"}
                    if args.pytorch else {})},
        "input_hashes": hashes["committed"],
        "candidates_sha256": hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
        "median_per_iteration_us": medians,
        "ratio_retained_over_committed": (medians["retained"] / medians["committed"]
                                          if medians["retained"] and medians["committed"] else None),
        "rows": rows,
    }
    (out / "capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("-" * 78)
    for arm in arms:
        values = ", ".join("%.3f" % row["per_iteration_us"] for row in rows if row["arm"] == arm)
        print("%-9s kernel time per iteration: %s | median %.3f us" % (arm, values, medians[arm]))
    if summary["ratio_retained_over_committed"]:
        print("retained / committed = %.4f in kernel time" % summary["ratio_retained_over_committed"])
    print("evidence", out / "capture.json")
    render.write_candidates(render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM,
                            render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM, generation=0)
    subprocess.run(["bash", "-lc", BUILD_COMMAND], cwd=REPO_ROOT,
                   capture_output=True, text=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
