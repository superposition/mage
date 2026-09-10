# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Render the committed evidence into responsive, downloadable scientific plots.

Run: uv run --script scripts/plot-comparison.py --experiment mage-001
The experiment selects the result and figure namespaces under docs/assets. It
defaults to mage-001 and can also be set through MAGE_EXPERIMENT.
The Pages build uses the committed SVGs and requires no plotting dependencies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mage-001"
SOURCE = ROOT / "docs/assets/results" / EXPERIMENT
OUT = ROOT / "docs/assets/figures" / EXPERIMENT
# The metadata sentence names the implementations a figure compares, so it
# travels with the experiment. mage-001 keeps its published wording.
BASELINE = {
    "mage-001": "PyTorch baseline versus first Triton and cuda-oxide Rust implementations.",
    "mage-002": "PyTorch and Triton baselines versus the rewritten cuda-oxide Rust kernels.",
}
LABELS = {"matmul": "Matrix multiplication", "gelu": "Bias + GELU", "layernorm": "LayerNorm",
          "triangle": "Triangle contraction", "neighbor": "Neighbor aggregation"}
LANGUAGES = ["python", "triton", "rust"]
DISPLAY = {"python": "PyTorch", "triton": "Triton", "rust": "Rust"}
COLORS = {"python": "#93caff", "triton": "#91dbba", "rust": "#c9b2ff"}
BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
METRICS = {
    "kernel": ("GPU kernel time", "Microseconds per operation · lower is better"),
    "event": ("Time around the call", "CUDA-event microseconds · lower is better"),
    "launches": ("Kernel launches", "Launches per operation"),
}


def baseline_note():
    return BASELINE.get(EXPERIMENT, "PyTorch and Triton baselines versus the cuda-oxide Rust kernels "
                                    f"published as {EXPERIMENT}.")


def read_data():
    events = json.loads((SOURCE / "comparison-results.json").read_text())
    profiles = json.loads((SOURCE / "comparison-profiles.json").read_text())
    indexed = {(r["operation"], r["language"]): r for r in profiles["captures"]}
    result = []
    for row in events["results"]:
        measurements = {}
        for lang in LANGUAGES:
            observations = [r["implementations"][lang] for r in row["rounds"]]
            samples = [x for r in observations for x in r["samples_us"]]
            assert len(samples) == events["rounds"] * events["iterations_per_round"]
            assert all(r["correctness"]["passed"] for r in observations)
            assert all(math.isfinite(x) and x >= 0 for x in samples)
            round_means = [statistics.mean(r["samples_us"]) for r in observations]
            capture = indexed[row["op"], lang]
            durations = [k["duration_us"] for k in capture["kernel_metrics"]]
            assert len(durations) == capture["launches"]
            assert all(math.isfinite(x) and x >= 0 for x in durations)
            launches = len(durations) / capture["requested_iterations"]
            assert launches.is_integer()
            measurements[lang] = {
                "event": statistics.mean(samples), "event_min": min(round_means), "event_max": max(round_means),
                "kernel": sum(durations) / capture["requested_iterations"], "launches": int(launches),
            }
        result.append({"op": row["op"], "label": LABELS[row["op"]], "dims": row["dims"],
                       "measurements": measurements})
    assert len(result) == 5
    return {"gpu": events["gpu"], "rounds": events["rounds"],
            "samples": events["rounds"] * events["iterations_per_round"], "rows": result,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (SOURCE / "comparison-results.json", SOURCE / "comparison-profiles.json")}}


def plot(data, metric, mobile):
    # Small multiples have independent zero-based axes so differences remain
    # legible for both ~8 us elementwise kernels and ~340 us matrix products.
    width, height = (3.65, 7.35) if mobile else (7.4, 6.5)
    fig, axes = plt.subplots(5, 1, figsize=(width, height), dpi=100)
    fig.subplots_adjust(left=.19 if mobile else .105, right=.97, top=.85, bottom=.075, hspace=.85)
    title, subtitle = METRICS[metric]
    fig.text(.01, .978, title, color=INK, fontsize=13 if mobile else 15, weight="bold", va="top")
    fig.text(.01, .943, subtitle, color=MUTED, fontsize=8.6 if mobile else 10, va="top")
    fig.text(.01, .012, "RTX 4090 · FP32 · scales differ by row", color=MUTED,
             fontsize=8 if mobile else 9, va="bottom")
    for ax, row in zip(axes, data["rows"]):
        values = [row["measurements"][l][metric] for l in LANGUAGES]
        extent = max(row["measurements"][l]["event_max"] if metric == "event" else values[i]
                     for i, l in enumerate(LANGUAGES))
        ax.set_xlim(0, extent * (1.31 if mobile else 1.19))
        ax.set_ylim(2.65, -.6)
        for y, (lang, value) in enumerate(zip(LANGUAGES, values)):
            ax.barh(y, value, height=.52, color=COLORS[lang], zorder=3)
            edge = value
            if metric == "event":
                low, high = (row["measurements"][lang][f"event_{s}"] for s in ("min", "max"))
                ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=INK,
                            capsize=2.5, linewidth=1, zorder=4)
                edge = high
            label = str(int(value)) if metric == "launches" else f"{value:.1f}"
            ax.annotate(label, (edge, y), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=9.5 if mobile else 11)
        ax.set_yticks(range(3), [DISPLAY[l] for l in LANGUAGES], color=MUTED, fontsize=8.5 if mobile else 10)
        ax.set_title(row["label"], loc="left", color=INK, fontsize=10.5 if mobile else 12, pad=6, weight="medium")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if mobile else 5, integer=metric == "launches"))
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8 if mobile else 9)
        ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    name = f"comparison-{metric}{'-mobile' if mobile else ''}"
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None, "Description":
        "RTX 4090; same FP32 inputs. Each row has its own zero-based scale. "
        f"{baseline_note()} "
        "See comparison-results.json and comparison-profiles.json for evidence."})
    svg = OUT / f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    if not mobile:
        fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


VIEWS = (("kernel", "GPU kernel time"), ("event", "Time around the call"))


def plot_views(data, mobile):
    """Both views in one figure: kernel time and time around the call, side by side.

    The reversal in Bias + GELU is the point of this figure, so the two measurements
    that disagree for the same operation sit next to each other.
    """
    if mobile:
        fig, axes = plt.subplots(len(data["rows"]) * len(VIEWS), 1, figsize=(3.65, 16.5), dpi=100)
        fig.subplots_adjust(left=.33, right=.97, top=.965, bottom=.028, hspace=1.15)
    else:
        fig, axes = plt.subplots(len(VIEWS), len(data["rows"]), figsize=(15.6, 7.4), dpi=100,
                                 gridspec_kw={"wspace": .55, "hspace": .52})
        fig.subplots_adjust(left=.032, right=.995, top=.855, bottom=.085)
    fig.patch.set_facecolor(BG)
    fig.text(.01, .985, "Kernel time and time around the call", color=INK,
             fontsize=12 if mobile else 15, weight="bold", va="top")
    fig.text(.01, .962 if mobile else .918, "The same five operations measured inside the kernel and around the call.",
             color=MUTED, fontsize=8.4 if mobile else 9.6, va="top")
    fig.text(.01, .006, "RTX 4090 · FP32 · each column has its own scale", color=MUTED,
             fontsize=8 if mobile else 9, va="bottom")
    for column, row in enumerate(data["rows"]):
        for panel, (metric, panel_title) in enumerate(VIEWS):
            ax = axes[column * len(VIEWS) + panel] if mobile else axes[panel][column]
            values = [row["measurements"][lang][metric] for lang in LANGUAGES]
            extent = max(row["measurements"][lang]["event_max"] if metric == "event" else values[i]
                         for i, lang in enumerate(LANGUAGES))
            ax.set_xlim(0, extent * (1.33 if mobile else 1.21))
            ax.set_ylim(2.65, -.6)
            for y, (lang, value) in enumerate(zip(LANGUAGES, values)):
                ax.barh(y, value, height=.52, color=COLORS[lang], zorder=3)
                edge = value
                if metric == "event":
                    low, high = (row["measurements"][lang][f"event_{s}"] for s in ("min", "max"))
                    ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=INK,
                                capsize=2.5, linewidth=1, zorder=4)
                    edge = high
                ax.annotate(f"{value:.1f}", (edge, y), xytext=(5, 0), textcoords="offset points",
                            ha="left", va="center", color=INK, fontsize=9.5 if mobile else 11)
            ax.set_yticks(range(3), [DISPLAY[lang] for lang in LANGUAGES], color=MUTED,
                          fontsize=8.5 if mobile else 10)
            ax.set_title(f"{row['label']} · {panel_title.lower()}" if mobile else row["label"],
                         loc="left", color=INK, fontsize=10 if mobile else 12, pad=6, weight="medium")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2 if mobile else 4))
            ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8 if mobile else 9)
            ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor(BG)
    name = f"comparison-views{'-mobile' if mobile else ''}"
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None, "Description":
        "RTX 4090; same FP32 inputs. GPU kernel time comes from a separate Nsight Systems capture; the "
        "event spans are means over three rotating rounds. Each column has its own zero-based scale. "
        "See comparison-results.json and comparison-profiles.json for evidence."})
    svg = OUT / f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    if not mobile:
        fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=os.environ.get("MAGE_EXPERIMENT", "mage-001"),
                        help="namespace under docs/assets/results and docs/assets/figures")
    args = parser.parse_args()
    EXPERIMENT = args.experiment
    SOURCE = ROOT / "docs/assets/results" / EXPERIMENT
    OUT = ROOT / "docs/assets/figures" / EXPERIMENT
    # Embed glyph outlines so downloads render identically without local fonts.
    # The page provides descriptive alt text and an accessible numeric table.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": f"{EXPERIMENT}-comparison"})
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_data()
    for metric in METRICS:
        for mobile in (False, True):
            plot(data, metric, mobile)
    for mobile in (False, True):
        plot_views(data, mobile)
    # mage-001 keeps the published data name its field note reads through site.data.
    name = ("profile_comparison.json" if EXPERIMENT == "mage-001"
            else f"profile_comparison_{EXPERIMENT.replace('-', '_')}.json")
    target = ROOT / "docs/_data" / name
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(data, indent=2))
