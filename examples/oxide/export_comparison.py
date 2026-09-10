"""Retain portable evidence for the public plots, without native report binaries.

The experiment names the result namespace and defaults to mage-001. Capture
directories are written relative to the checkout that produced them, so the
committed JSON carries no workstation paths.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def portable(report_dir, roots):
    """Rewrite a capture directory as a path relative to the first containing root.

    The committed JSON is read away from the machine that produced it, so an
    absolute path there would expose one workstation's layout and resolve nowhere.
    """
    path = Path(report_dir)
    for root in roots:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def export(events, profiles, destination, experiment):
    destination.mkdir(parents=True, exist_ok=True)
    results = json.loads((events / "results.json").read_text())
    results["experiment_id"] = f"{experiment}-comparison"
    (destination / "comparison-results.json").write_text(json.dumps(results, indent=2) + "\n")
    captures = json.loads((profiles / "summary.json").read_text())
    for capture in captures:
        metrics = json.loads((Path(capture["report_dir"]) / "kernels.json").read_text())
        if len(metrics) != capture["launches"] or capture["status"] != "complete":
            raise ValueError("Incomplete capture")
        capture["report_dir"] = portable(capture["report_dir"], (ROOT, profiles))
        capture["kernel_metrics"] = [
            {key: metric[key] for key in ("kernel_name", "duration_us", "grid_size", "block_size",
                                         "registers_per_thread", "shared_mem_bytes")}
            for metric in metrics]
    data = {"experiment_id": f"{experiment}-comparison", "backend": "nsys",
            "timestamp_mode": "WSL CuptiUseRawGpuTimestamps=false (reduced timestamp precision)",
            "timing": "sum of CUDA kernel durations / requested iterations; excludes gaps; separate profiled run",
            "captures": captures}
    (destination / "comparison-profiles.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--experiment", default="mage-001", help="result namespace, e.g. mage-001 or mage-002")
    parser.add_argument("--destination", type=Path,
                        help="defaults to docs/assets/results/<experiment>")
    args = parser.parse_args()
    export(args.events, args.profiles, args.destination or ROOT / "docs/assets/results" / args.experiment,
           args.experiment)
