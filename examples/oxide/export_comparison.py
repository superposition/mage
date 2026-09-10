"""Retain portable evidence for the public plots, without native report binaries."""
import argparse
import json
from pathlib import Path
import shutil


def export(events, profiles, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(events / "results.json", destination / "comparison-results.json")
    captures = json.loads((profiles / "summary.json").read_text())
    for capture in captures:
        metrics = json.loads((Path(capture["report_dir"]) / "kernels.json").read_text())
        if len(metrics) != capture["launches"] or capture["status"] != "complete":
            raise ValueError("Incomplete capture")
        capture["kernel_metrics"] = [
            {key: metric[key] for key in ("kernel_name", "duration_us", "grid_size", "block_size",
                                         "registers_per_thread", "shared_mem_bytes")}
            for metric in metrics]
    data = {"experiment_id": "mage-001-comparison", "backend": "nsys",
            "timestamp_mode": "WSL CuptiUseRawGpuTimestamps=false (reduced timestamp precision)",
            "timing": "sum of CUDA kernel durations / requested iterations; excludes gaps; separate profiled run",
            "captures": captures}
    (destination / "comparison-profiles.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("docs/assets/results/mage-001"))
    args = parser.parse_args()
    export(args.events, args.profiles, args.destination)
