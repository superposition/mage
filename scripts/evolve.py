#!/usr/bin/env python3
"""Closed measurement -> proposal -> validation loop for the mage-oxide kernels.

Each generation renders `examples/oxide/src/candidates.rs` from the knobs in
`evolve_render`, builds it, then measures three arms in fresh processes: the
generated `best` (incumbent), the generated `new` (proposal), and the committed
kernels (a manifest with no `variant` key). The candidate is gated on
full-output correctness against the PyTorch FP32 reference before any timing is
judged, the committed arm bounds the session's drift, and every generation is
appended to `ledger.jsonl` whether it was accepted or not.

Measurement boundary: every number this loop compares is a CUDA event span
recorded around one launch call, so it includes host-side launch work and
excludes transfer, allocation, and process startup. It is not GPU kernel
duration.

Exit codes: 0 once at least one measurement round completed, 2 for a setup
failure (missing template, missing binary, unusable inputs), 3 when setup
succeeded but no round was ever measured.

Integration: the build and the run are the only steps that touch the build tree,
and each is isolated in one function (`build_candidates`, `measure_variant`).
`EVOLVE_BUILD_COMMAND` and `EVOLVE_BINARY` override them from the environment so
the loop can be driven without a rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import evolve_policy as policy  # noqa: E402
import evolve_render as render  # noqa: E402

EXPERIMENT = REPO_ROOT / "examples" / "oxide" / "experiment.py"
# Integration seams: the build and the run are the only steps that touch the
# build tree. Each is one function, and each is overridable from the environment
# so the loop can be exercised, or integrated, without rebuilding the tree.
BINARY = Path(os.environ.get(
    "EVOLVE_BINARY",
    str(REPO_ROOT / "examples" / "oxide" / "target" / "release" / "mage-oxide"),
))
BUILD_COMMAND = os.environ.get(
    "EVOLVE_BUILD_COMMAND",
    "source scripts/oxide-env.sh && cd examples/oxide && "
    "CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89",
)
BUILD_TIMEOUT = 900
RUN_TIMEOUT = 600
RESERVED_PREFIXES = ("mage-", "kernel-", "python-", "rust-")
DEFAULT_SHAPES = {"matmul": (1024, 1024, 1024), "layernorm": (4096, 768)}
VARIANTS = ("best", "new", "committed")


# --- small helpers ---------------------------------------------------------


def setup_failure(message: str) -> None:
    """Abort before any generation: exit 2 with the reason on stderr."""
    print(f"setup failure: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def load_experiment():
    spec = importlib.util.spec_from_file_location("evolve_experiment", EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_shape(op: str, raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_SHAPES[op]
    parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
    try:
        dims = tuple(int(piece) for piece in parts)
    except ValueError as error:
        raise SystemExit(f"--shape must be comma-separated integers: {raw!r}") from error
    expected = 3 if op == "matmul" else 2
    if len(dims) != expected:
        raise SystemExit(f"--shape for {op} needs {expected} values, got {dims}")
    if any(dim <= 0 for dim in dims):
        raise SystemExit(f"--shape must be positive: {dims}")
    return dims


def template_hashes() -> dict[str, str]:
    hashes = {}
    for relative in render.READ_ONLY_TEMPLATES:
        path = REPO_ROOT / relative
        if not path.is_file():
            setup_failure(f"missing template {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def repo_revision() -> str:
    result = run_command(["git", "rev-parse", "HEAD"], timeout=60)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def gpu_name() -> str:
    result = run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=60
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    return "unavailable"


# --- measurement plumbing --------------------------------------------------


def install_inputs(source: Path, dest: Path, variant: str) -> None:
    """Copy the fixed inputs into a fresh directory and select the variant."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    manifest_path = dest / "input.json"
    manifest = json.loads(manifest_path.read_text())
    if variant == "committed":
        manifest.pop("variant", None)
    else:
        manifest["variant"] = variant
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def measure_variant(directory: Path, variant: str) -> dict:
    """Run one arm in its own process and read back its timing record."""
    started = time.monotonic()
    try:
        process = run_command([str(BINARY), str(directory)], timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"variant": variant, "error": f"run timed out after {RUN_TIMEOUT}s"}
    record = {
        "variant": variant,
        "returncode": process.returncode,
        "seconds": round(time.monotonic() - started, 3),
    }
    if process.returncode != 0:
        record["error"] = (process.stderr or process.stdout or "no output").strip()[-2000:]
        return record
    timing_path = directory / "rust-timing.json"
    if not timing_path.is_file():
        record["error"] = "binary exited 0 but wrote no rust-timing.json"
        return record
    timing = json.loads(timing_path.read_text())
    samples = timing.get("samples_us")
    if not samples:
        record["error"] = "rust-timing.json carried no samples_us"
        return record
    record.update(
        median_us=float(statistics.median(samples)),
        mean_us=float(timing["event_mean_us"]),
        run_id=timing.get("run_id"),
        samples=len(samples),
    )
    return record


def build_candidates() -> dict:
    """Build the rendered candidates; the only step that mutates the build tree."""
    started = time.monotonic()
    try:
        build = run_command(["bash", "-c", BUILD_COMMAND], timeout=BUILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": round(time.monotonic() - started, 3),
                "error": f"build timed out after {BUILD_TIMEOUT}s"}
    return {
        "ok": build.returncode == 0,
        "seconds": round(time.monotonic() - started, 3),
        "error": None if build.returncode == 0 else (build.stdout + build.stderr)[-4000:],
    }


def correctness_gate(experiment, directory: Path) -> dict:
    """Full-output comparison against the PyTorch reference, verbatim on failure."""
    try:
        result = experiment.validate(directory)
    except AssertionError as error:
        return {"passed": False, "max_abs_error": None,
                "detail": error.args[0] if error.args else str(error)}
    except Exception as error:  # validation failure must not abort the loop
        return {"passed": False, "max_abs_error": None,
                "detail": f"{type(error).__name__}: {error}"}
    return {
        "passed": bool(result.get("passed")),
        "max_abs_error": result.get("max_abs_error"),
        "checked_elements": result.get("checked_elements"),
    }


def round_order(base: list[str], round_index: int) -> list[str]:
    offset = round_index % len(base)
    return base[offset:] + base[:offset]


# --- argument parsing ------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--op", choices=sorted(DEFAULT_SHAPES), default="matmul")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--shape", default=None,
                        help="m,n[,k] for matmul; rows,width for layernorm")
    parser.add_argument("--max-stale", type=int, default=6,
                        help="stop after this many consecutive rejected generations")
    parser.add_argument("--min-gain", type=float, default=0.01,
                        help="relative improvement required to accept a candidate")
    parser.add_argument("--control-drift", type=float, default=0.03,
                        help="tolerated relative drift of the committed control (max/min - 1)")
    parser.add_argument("--out", default=None,
                        help="result namespace (default artifacts/evolution/<utc>_<op>)")
    parser.add_argument("--seed", type=int, default=20260910,
                        help="determinism for input generation and measurement-order ties")
    parser.add_argument("--start-json", default=None,
                        help="JSON object overriding the starting knobs for --op")
    parser.add_argument("--work-dir", default=None,
                        help="generated inputs and per-round measurement directories")
    args = parser.parse_args(argv)
    if not 1 <= args.generations <= 500:
        raise SystemExit("--generations must be between 1 and 500")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--iterations must be positive and --warmup non-negative")
    return args


def resolve_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{args.op}"
    if args.out is None:
        out = (REPO_ROOT / "artifacts" / "evolution" / run_id).resolve()
    else:
        out = Path(args.out).resolve()
        run_id = out.name or run_id
    if out.name.startswith(RESERVED_PREFIXES):
        raise SystemExit(f"refusing to reuse the reserved result namespace {out.name!r}")
    if (out / "ledger.jsonl").exists() or (out / "summary.json").exists():
        raise SystemExit(f"refusing to overwrite existing evidence in {out}")
    work = Path(args.work_dir).resolve() if args.work_dir else out / "work"
    return run_id, out, work


# --- the loop --------------------------------------------------------------


def repair_with_staging(op: str, params: dict, dims) -> dict | None:
    """Pair an otherwise-invalid geometry proposal with the staging mode that fits it.

    A guard-free incumbent (staging: "shared") forbids every tile and k-step
    move, and switching modes on its own is a within-noise edit, so coordinate
    descent would never reach the rest of the space. When the validator rejects a
    proposal, retrying it with each remaining mode turns a predicted dead end into
    one measurable candidate. Failure is a rejection exactly as before; the repair
    is recorded in the ledger so the pairing stays visible.
    """
    try:
        render.validate(op, params, dims)
        return None
    except ValueError:
        pass
    if op != "matmul":
        return None
    current = str(params.get("staging"))
    for mode in ("exact", "guarded"):
        if mode == current:
            continue
        candidate = dict(params)
        candidate["staging"] = mode
        try:
            render.validate(op, candidate, dims)
        except ValueError:
            continue
        return {"knob": "staging", "from": current, "to": mode, "params": candidate}
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dims = parse_shape(args.op, args.shape)
    run_id, out, work = resolve_paths(args)
    out.mkdir(parents=True, exist_ok=True)

    if not BINARY.is_file():
        setup_failure(f"missing binary {BINARY} (build it first)")
    templates = template_hashes()
    revision = repo_revision()

    try:
        experiment = load_experiment()
    except Exception as error:
        setup_failure(f"cannot import {EXPERIMENT}: {error}")
    torch_version = getattr(experiment.torch, "__version__", "unknown")

    print(f"run          {run_id}")
    print(f"repo         {REPO_ROOT} @ {revision}")
    print(f"gpu          {gpu_name()}")
    print(f"torch        {torch_version}")
    print(f"op/shape     {args.op} {list(dims)}")
    print(f"budget       {args.generations} generations x {args.rounds} rounds x "
          f"{args.iterations} iterations (warmup {args.warmup})")
    print("boundary     CUDA event span around one launch call (host launch work included, "
          "GPU kernel duration not isolated)")
    print("-" * 100, flush=True)

    # --- per-run setup: fixed inputs, reference, provenance ---------------
    inputs_dir = work / "inputs" / args.op
    try:
        experiment.generate(
            inputs_dir, args.op, list(dims),
            seed=args.seed, warmup=args.warmup, iterations=args.iterations,
        )
        _, reference_fn = experiment.reference(inputs_dir)
        reference_fn()
        experiment.torch.cuda.synchronize()
    except Exception as error:
        setup_failure(f"cannot generate/load inputs: {error}")

    input_hashes = {
        name: sha256_file(inputs_dir / name)
        for name in ("a.bin", "b.bin", "c.bin")
        if (inputs_dir / name).is_file()
    }
    if not (inputs_dir / "inputs.sha256.json").is_file():
        (inputs_dir / "inputs.sha256.json").write_text(
            json.dumps({"seed": args.seed, "sha256": input_hashes}, indent=2) + "\n"
        )
    (out / "inputs.sha256.json").write_text(
        json.dumps({
            "op": args.op, "dims": list(dims), "seed": args.seed,
            "warmup": args.warmup, "iterations": args.iterations,
            "sha256": input_hashes,
            "input_json_sha256": sha256_file(inputs_dir / "input.json"),
        }, indent=2) + "\n"
    )

    incumbents = {"matmul": dict(render.DEFAULT_MATMUL),
                  "layernorm": dict(render.DEFAULT_LAYERNORM)}
    if args.start_json:
        try:
            override = json.loads(args.start_json)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--start-json is not valid JSON: {error}") from error
        if not isinstance(override, dict):
            raise SystemExit("--start-json must be a JSON object of knobs")
        unknown = sorted(set(override) - set(incumbents[args.op]))
        if unknown:
            raise SystemExit(f"--start-json has unknown {args.op} knobs: {', '.join(unknown)}")
        incumbents[args.op].update(override)
    try:
        render.validate(args.op, incumbents[args.op], dims)
    except ValueError as error:  # ParameterError included: the start config must be buildable
        setup_failure(f"starting configuration is not buildable: {error}")
    start_params = dict(incumbents[args.op])
    knobs = render.MATMUL_KNOBS if args.op == "matmul" else render.LAYERNORM_KNOBS
    proposer = policy.Proposer(knobs, start_params, seed=args.seed)

    base_order = list(VARIANTS)
    random.Random(args.seed).shuffle(base_order)

    ledger_path = out / "ledger.jsonl"
    entries: list[dict] = []
    accepted = rejected = 0
    noisy_generations = 0
    rounds_completed = 0
    stale = 0
    stop_reason = "reached generation budget"
    start_median_us: float | None = None
    # Which candidates.rs slot held the incumbent at the last successful build.
    # The confirmation round must read that slot: after an accepted generation
    # the incumbent lives in `new`, not `best`.
    incumbent_slot = "best"

    def render_candidates(generation: int, new_params: dict) -> str:
        new_matmul = new_params if args.op == "matmul" else incumbents["matmul"]
        new_layernorm = new_params if args.op == "layernorm" else incumbents["layernorm"]
        return render.write_candidates(
            incumbents["matmul"], incumbents["layernorm"], new_matmul, new_layernorm,
            generation=generation, dest=render.CANDIDATES_PATH,
        )

    def append_ledger(entry: dict) -> None:
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    for generation in range(1, args.generations + 1):
        generation_started = time.monotonic()
        try:
            proposal = proposer.propose()
        except Exception as error:  # the policy must never take the loop down
            stop_reason = f"proposal policy failed: {type(error).__name__}: {error}"
            print(f"evolve: {stop_reason}", file=sys.stderr)
            break
        if proposal is None:
            stop_reason = "all knobs exhausted under the current incumbent"
            break
        new_params = dict(proposal.params)
        repair = repair_with_staging(args.op, new_params, dims)
        if repair is not None:
            new_params = dict(repair["params"])
        best_params = dict(incumbents[args.op])
        entry = {
            "generation": generation,
            "ts": utc_now(),
            "op": args.op,
            "shape": list(dims),
            "best_params": best_params,
            "new_params": new_params,
            "changed_knob": proposal.knob,
            "changed_from": proposal.before,
            "changed_to": proposal.after,
            "hypothesis": proposal.hypothesis,
            "build": {"ok": False, "seconds": 0.0, "error": None},
            "rounds": [],
            "correctness": {"passed": False, "max_abs_error": None},
            "control_drift": None,
            "noisy": False,
            "decision": "reject",
            "reason": "",
            "best_before": best_params,
            "best_after": best_params,
            "elapsed_seconds": 0.0,
            "candidates_sha256": None,
            "templates": templates,
            "repo_revision": revision,
        }
        if repair is not None:
            entry["repair"] = {key: value for key, value in repair.items() if key != "params"}

        def round_medians(variant: str) -> list[float]:
            key = f"{variant}_median_us"
            return [row[key] for row in entry["rounds"] if row.get(key) is not None]

        try:
            # (b) validate both roles, then render both roles
            try:
                render.validate("matmul",
                                best_params if args.op == "matmul" else incumbents["matmul"],
                                dims if args.op == "matmul" else None)
                render.validate("layernorm",
                                best_params if args.op == "layernorm" else incumbents["layernorm"],
                                dims if args.op == "layernorm" else None)
                render.validate(args.op, new_params, dims)
            except ValueError as error:
                entry["build"]["error"] = f"rejected by evolve_render.validate: {error}"
                entry["reason"] = str(error)
                entry["elapsed_seconds"] = round(time.monotonic() - generation_started, 3)
                append_ledger(entry)
                entries.append(entry)
                rejected += 1
                stale += 1
                proposer.record(proposal, accepted=False)
                print(f"gen {generation:>3} | {proposal.knob} "
                      f"{policy.format_value(proposal.before)}->{policy.format_value(proposal.after)}"
                      f" | invalid proposal | REJECT ({str(error)[:70]})", flush=True)
                if stale >= args.max_stale:
                    stop_reason = f"{stale} consecutive rejections"
                    break
                continue

            entry["candidates_sha256"] = render_candidates(generation, new_params)

            # (c) build; a bad candidate must never take the loop down
            entry["build"] = build_candidates()
            if not entry["build"]["ok"]:
                entry["reason"] = "candidate failed to build; the incumbent binary is untouched"
                entry["elapsed_seconds"] = round(time.monotonic() - generation_started, 3)
                append_ledger(entry)
                entries.append(entry)
                rejected += 1
                stale += 1
                proposer.record(proposal, accepted=False)
                print(f"gen {generation:>3} | {proposal.knob} "
                      f"{policy.format_value(proposal.before)}->{policy.format_value(proposal.after)}"
                      f" | build failed | REJECT", flush=True)
                if stale >= args.max_stale:
                    stop_reason = f"{stale} consecutive rejections"
                    break
                continue

            # (d) measure every arm, rotating the order per round
            for round_index in range(args.rounds):
                order = round_order(base_order, round_index)
                row = {"round": round_index + 1, "order": list(order)}
                runs = []
                for variant in order:
                    directory = (work / "measure" / f"g{generation:03d}"
                                 / f"r{round_index + 1}" / variant)
                    install_inputs(inputs_dir, directory, variant)
                    record = measure_variant(directory, variant)
                    runs.append(record)
                    row[f"{variant}_median_us"] = record.get("median_us")
                    row[f"{variant}_mean_us"] = record.get("mean_us")
                    row[f"{variant}_run_id"] = record.get("run_id")
                    if record.get("median_us") is not None:
                        rounds_completed += 1
                    else:
                        row.setdefault("errors", []).append(
                            f"{variant}: {record.get('error', 'no timing')[:200]}"
                        )
                row["runs"] = runs
                entry["rounds"].append(row)

            run_errors = [error for row in entry["rounds"] for error in row.get("errors", [])]
            if run_errors:
                entry["reason"] = "; ".join(run_errors)[:400]

            # (e) correctness gate on the first best/new outputs
            if not run_errors:
                new_check = correctness_gate(
                    experiment, work / "measure" / f"g{generation:03d}" / "r1" / "new")
                best_check = correctness_gate(
                    experiment, work / "measure" / f"g{generation:03d}" / "r1" / "best")
                entry["correctness"] = {
                    "passed": bool(new_check["passed"] and best_check["passed"]),
                    "max_abs_error": new_check.get("max_abs_error"),
                    "new": new_check,
                    "best": best_check,
                }

            # (f) decision
            verdict = policy.decide(
                correctness_passed=entry["correctness"]["passed"] and not run_errors,
                new_medians=round_medians("new"),
                best_medians=round_medians("best"),
                committed_medians=round_medians("committed"),
                min_gain=args.min_gain,
                control_tolerance=args.control_drift,
            )
            entry["decision"] = verdict["decision"]
            entry["noisy"] = verdict["noisy"]
            entry["control_drift"] = verdict["control_drift"]
            if not run_errors:
                entry["reason"] = verdict["reason"]
            if start_median_us is None and accepted == 0 and round_medians("best"):
                # The first measured incumbent arm is the start configuration; a
                # generation rejected before it ever measured contributes nothing.
                start_median_us = statistics.median(round_medians("best"))

            if verdict["decision"] == "accept":
                proposer.record(proposal, accepted=True, gain=1.0 - verdict["ratio"])
                incumbents[args.op] = new_params
                incumbent_slot = "new"
                accepted += 1
                stale = 0
            else:
                proposer.record(proposal, accepted=False)
                incumbent_slot = "best"
                rejected += 1
                stale += 1
            if verdict["noisy"]:
                noisy_generations += 1
        except Exception as error:  # keep the loop alive and the render consistent
            entry["decision"] = "reject"
            entry["reason"] = f"{type(error).__name__}: {error}"
            rejected += 1
            stale += 1
            try:
                render_candidates(generation, dict(incumbents[args.op]))
            except Exception as restore_error:
                print(f"warning: could not restore candidates.rs: {restore_error}",
                      file=sys.stderr)

        entry["best_after"] = dict(incumbents[args.op])
        entry["elapsed_seconds"] = round(time.monotonic() - generation_started, 3)
        append_ledger(entry)
        entries.append(entry)

        # Same statistic the decision uses (the fastest observed round on each arm),
        # so the printed line can never disagree with the recorded verdict.
        best_median = min(round_medians("best")) if round_medians("best") else None
        new_median = min(round_medians("new")) if round_medians("new") else None
        ratio = (new_median / best_median) if (new_median and best_median) else None
        print(
            f"gen {generation:>3} | {proposal.knob} "
            f"{policy.format_value(proposal.before)}->{policy.format_value(proposal.after)}"
            f" | best {best_median if best_median is not None else float('nan'):8.2f}us"
            f" new {new_median if new_median is not None else float('nan'):8.2f}us"
            f" | ratio {ratio if ratio is not None else float('nan'):.3f}"
            f" | {entry['decision'].upper()}"
            + (" (noisy)" if entry["noisy"] else ""),
            flush=True,
        )
        if stale >= args.max_stale:
            stop_reason = f"{stale} consecutive rejections"
            break

    # --- end of run: fresh-process confirmation ---------------------------
    confirmation: list[dict] = []
    try:
        for round_index in range(args.rounds):
            # The final incumbent lives in whichever slot the last successful
            # build rendered it into, so that slot is what the confirmation reads.
            order = [variant for variant in round_order(base_order, round_index)
                     if variant in (incumbent_slot, "committed")]
            row = {"round": round_index + 1, "order": list(order),
                   "incumbent_variant": incumbent_slot}
            for variant in order:
                directory = work / "confirm" / f"r{round_index + 1}" / variant
                install_inputs(inputs_dir, directory, variant)
                record = measure_variant(directory, variant)
                row[f"{variant}_median_us"] = record.get("median_us")
                row[f"{variant}_mean_us"] = record.get("mean_us")
                row[f"{variant}_run_id"] = record.get("run_id")
                if record.get("median_us") is None:
                    row[f"{variant}_error"] = record.get("error")
                    continue
                if variant == incumbent_slot:
                    row["incumbent_median_us"] = record["median_us"]
                    row["incumbent_mean_us"] = record["mean_us"]
            confirmation.append(row)
    except Exception as error:
        print(f"warning: confirmation rounds incomplete: {error}", file=sys.stderr)

    def confirmation_medians(key: str) -> list[float]:
        return [row[key] for row in confirmation if row.get(key) is not None]

    best_medians = confirmation_medians("incumbent_median_us")
    committed_medians = confirmation_medians("committed_median_us")
    best_median_us = statistics.median(best_medians) if best_medians else None
    committed_median_us = statistics.median(committed_medians) if committed_medians else None
    improvement_vs_committed = (
        1.0 - best_median_us / committed_median_us
        if best_median_us and committed_median_us else None
    )
    improvement_vs_start = (
        1.0 - best_median_us / start_median_us
        if best_median_us and start_median_us else None
    )

    summary = {
        "run_id": run_id,
        "op": args.op,
        "shape": list(dims),
        "generations_run": len(entries),
        "accepted": accepted,
        "rejected": rejected,
        "final_best_params": dict(incumbents[args.op]),
        "start_params": start_params,
        "committed_median_us": committed_median_us,
        "best_median_us": best_median_us,
        "start_median_us": start_median_us,
        "improvement_vs_committed": improvement_vs_committed,
        "improvement_vs_start": improvement_vs_start,
        "knob_payoffs": proposer.ordered_payoffs(),
        "rejected_variants": [
            {"generation": entry["generation"], "params": entry["new_params"],
             "changed_knob": entry["changed_knob"], "reason": entry["reason"]}
            for entry in entries if entry["decision"] != "accept"
        ],
        "noisy_generations": [entry["generation"] for entry in entries if entry["noisy"]],
        "confirmation": confirmation,
        "incumbent_variant": incumbent_slot,
        "last_build_ok": bool(entries and entries[-1]["build"]["ok"]),
        "rounds_measured": rounds_completed,
        "stopped_after": stop_reason,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "min_gain": args.min_gain,
        "control_drift_tolerance": args.control_drift,
        "seed": args.seed,
        "repo_revision": revision,
        "candidates_sha256": entries[-1]["candidates_sha256"] if entries else None,
        "templates": templates,
        "inputs": input_hashes,
        "utc": utc_now(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def fmt(value: float | None) -> str:
        return f"{value:.2f}us" if value is not None else "n/a"

    print("-" * 100)
    print(f"generations  {summary['generations_run']} run, {accepted} accepted, {rejected} rejected"
          + (f", {noisy_generations} noisy" if noisy_generations else "")
          + f" | stopped: {stop_reason}")
    print(f"start params {json.dumps(start_params, sort_keys=True)}")
    print(f"final params {json.dumps(summary['final_best_params'], sort_keys=True)}")
    print(f"confirmed    {incumbent_slot} {fmt(best_median_us)} vs committed "
          f"{fmt(committed_median_us)}"
          + (f" | improvement {improvement_vs_committed:+.2%}"
             if improvement_vs_committed is not None else "")
          + (f" | vs start {improvement_vs_start:+.2%}"
             if improvement_vs_start is not None else ""))
    print(f"ledger       {ledger_path}")
    print(f"summary      {out / 'summary.json'}")

    if rounds_completed == 0:
        print("no measurement round ever completed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as error:
        if isinstance(error.code, str):
            print(f"evolve: {error.code}", file=sys.stderr)
        raise
