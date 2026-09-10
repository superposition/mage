"""Capture all five implementations, serially, through Mage's native backends."""
import argparse
import json
from pathlib import Path
import sys

import torch
from mage.profiler.backends import NcuBackend, NsysBackend
from mage.profiler.models import ProfileSession
from mage.profiler.storage import ProfileDB
from experiment import DEFAULTS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=["nsys", "ncu"], default="nsys")
parser.add_argument("--inputs", type=Path, default=Path("artifacts/mage-001"))
parser.add_argument("--output", type=Path, default=Path("artifacts/mage-001-profiles"))
parser.add_argument("--iterations", type=int)
parser.add_argument("--languages", nargs="+", choices=["rust", "python", "triton"], default=["rust", "python"])
args = parser.parse_args()
iterations = args.iterations or (1 if args.backend == "ncu" else 100)
if iterations <= 0:
    parser.error("iterations must be positive")
args.output.mkdir(parents=True, exist_ok=True)
db = ProfileDB(args.output / "profiles.db")
base = Path(__file__).resolve().parent
summaries = []
for op in DEFAULTS:
    for language in args.languages:
        target = args.inputs.resolve() / op
        if language == "rust":
            argv = [str(base / "target/release/mage-oxide"), str(target),
                    "--iterations", str(iterations), "--capture"]
        else:
            script = "triton_target.py" if language == "triton" else "python_target.py"
            argv = [sys.executable, str(base / script), str(target),
                    "--iterations", str(iterations)]
        cls = NcuBackend if args.backend == "ncu" else NsysBackend
        backend = cls(capture_range="cuda", output_dir=args.output / op / language,
                      launch_count=1 if args.backend == "ncu" else None)
        try:
            metrics = list(backend.run_command(argv))
            session = ProfileSession(command=__import__("shlex").join(argv), backend=args.backend,
                                     device_name=torch.cuda.get_device_name(0), metrics=metrics)
            session.finish()
            session_id = db.save_session(session)
            row = {"operation": op, "language": language, "backend": args.backend,
                   "launches": len(metrics), "requested_iterations": iterations,
                   "kernel_names": sorted({m.kernel_name for m in metrics}),
                   "session_id": session_id, "report_dir": str(backend.report_dir),
                   "status": "complete"}
            if language in {"rust", "triton"} and args.backend == "nsys" and len(metrics) != iterations:
                raise AssertionError(f"Expected {iterations} native launches, got {len(metrics)}")
            summaries.append(row)
            print(json.dumps(row), flush=True)
        finally:
            backend.cleanup()
(args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
