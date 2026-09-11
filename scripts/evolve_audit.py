"""Verify that every configuration the loop can reach is correct.

The loop's decisions are only meaningful if the kernels it generates are right: a
configuration that computes the wrong answer would be measured, compared, and possibly
kept. This renders each reachable configuration, builds it, runs it, and puts its full
output through the PyTorch FP32 reference, then reports one line per configuration.

Run with the Mage Python environment, from the repository root, with the cuda-oxide
toolchain available (see docs/guide.md):

    .venv/bin/python scripts/evolve_audit.py
    .venv/bin/python scripts/evolve_audit.py --op layernorm --shape 64,768

Exits non-zero if any configuration fails to build, fails to run, or disagrees with the
reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = REPO_ROOT / "examples" / "oxide" / "src" / "candidates.rs"
BINARY = REPO_ROOT / "examples" / "oxide" / "target" / "release" / "mage-oxide"
BUILD_COMMAND = (
    "source scripts/oxide-env.sh && cd examples/oxide && "
    "CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89"
)
DEFAULT_SHAPES = {"matmul": (256, 256, 128), "layernorm": (64, 768)}


def load(name: str, base: Path):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = load("evolve_render", REPO_ROOT / "scripts")
experiment = load("experiment", REPO_ROOT / "examples" / "oxide")


def matmul_space() -> list[dict]:
    """The reachable matmul configurations, in the renderer's own terms."""
    space = []
    for block in [(64, 64), (32, 64), (64, 32), (128, 64)]:
        for transpose_a in (True, False):
            for quad_stage in (True, False):
                for k_step in (32, 64):
                    for staging in render.STAGING_MODES:
                        params = dict(render.DEFAULT_MATMUL, block=block, transpose_a=transpose_a,
                                      quad_stage=quad_stage, k_step=k_step, staging=staging)
                        try:
                            render.validate("matmul", params)
                        except render.ParameterError:
                            continue
                        space.append(params)
    return space


def layernorm_space() -> list[dict]:
    space = []
    for warps_per_row in render.LAYERNORM_KNOBS["warps_per_row"]:
        for threads in render.LAYERNORM_KNOBS["threads"]:
            params = {"warps_per_row": warps_per_row, "threads": threads}
            try:
                render.validate("layernorm", params)
            except render.ParameterError:
                continue
            space.append(params)
    return space


def label(role: str, params: dict) -> str:
    if role == "matmul":
        block_m, block_n = params["block"]
        return ("b%dx%d_%s_%s_k%d_%s" % (
            block_m, block_n,
            "T" if params["transpose_a"] else "F",
            "Q" if params["quad_stage"] else "S",
            params["k_step"], params["staging"],
        ))
    return "wpr%d_t%d" % (params["warps_per_row"], params["threads"])


def build() -> tuple[bool, float, str | None]:
    started = datetime.now(timezone.utc)
    process = subprocess.run(["bash", "-lc", BUILD_COMMAND], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    if process.returncode == 0:
        return True, seconds, None
    lines = [line for line in (process.stdout + process.stderr).splitlines()
             if "error" in line.lower()]
    return False, seconds, (lines[0][:200] if lines else "build failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--op", choices=["matmul", "layernorm", "both"], default="both")
    parser.add_argument("--shape", default=None, help="m,n[,k] or rows,width")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out).resolve() if args.out else (
        REPO_ROOT / "artifacts" / "evolution" / "audit-%s" % stamp)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if not BINARY.is_file():
        raise SystemExit("setup failure: build the example first (%s)" % BUILD_COMMAND)

    roles = ["matmul", "layernorm"] if args.op == "both" else [args.op]
    records: list[dict] = []
    failures = 0
    for role in roles:
        if args.shape:
            dims = tuple(int(part) for part in args.shape.split(","))
        else:
            dims = DEFAULT_SHAPES[role]
        space = matmul_space() if role == "matmul" else layernorm_space()
        print("== %s: %d reachable configurations at %s ==" % (role, len(space), list(dims)))
        for params in space:
            name = label(role, params)
            record = {"role": role, "dims": list(dims), "params": params, "label": name}
            if role == "matmul":
                best_matmul, new_matmul = render.DEFAULT_MATMUL, params
                best_layernorm = new_layernorm = render.DEFAULT_LAYERNORM
            else:
                best_matmul = new_matmul = render.DEFAULT_MATMUL
                best_layernorm, new_layernorm = render.DEFAULT_LAYERNORM, params
            render.write_candidates(best_matmul, best_layernorm, new_matmul, new_layernorm,
                                    generation=0, dest=CANDIDATES)
            ok, seconds, error = build()
            record["build"] = {"ok": ok, "seconds": round(seconds, 2), "error": error}
            if not ok:
                failures += 1
                record["passed"] = False
                records.append(record)
                print("  %-28s BUILD FAIL  %s" % (name, error))
                continue
            directory = work / name
            experiment.generate(directory, role, list(dims), warmup=2,
                                iterations=args.iterations)
            manifest = json.loads((directory / "input.json").read_text())
            manifest["variant"] = "new"
            (directory / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
            run = subprocess.run([str(BINARY), str(directory)], capture_output=True, text=True)
            record["run"] = {"ok": run.returncode == 0, "stderr": run.stderr[-400:]}
            if run.returncode != 0:
                failures += 1
                record["passed"] = False
                records.append(record)
                print("  %-28s RUN FAIL    %s" % (name, run.stderr.strip()[:90]))
                continue
            try:
                correctness = experiment.validate(directory)
                record.update(passed=bool(correctness["passed"]),
                              max_abs_error=correctness["max_abs_error"])
                if not correctness["passed"]:
                    failures += 1
                print("  %-28s %s max_err=%.3g" % (
                    name, "correct" if correctness["passed"] else "WRONG  ",
                    correctness["max_abs_error"]))
            except AssertionError:
                record.update(passed=False, max_abs_error=None)
                failures += 1
                print("  %-28s WRONG    (reference mismatch)" % name)
            records.append(record)

    render.write_candidates(render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM,
                            render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM, generation=0)
    ok, seconds, error = build()
    if not ok:
        print("warning: could not restore the incumbent render: %s" % error, file=sys.stderr)

    summary = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configurations": len(records),
        "failures": failures,
        "templates": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                      for name in render.READ_ONLY_TEMPLATES},
        "records": records,
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("-" * 78)
    print("checked %d configurations, %d failed | evidence %s" % (len(records), failures, out))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
