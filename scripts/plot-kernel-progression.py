# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Draw how the two rewritten Rust kernels reached their measured kernel time.

Run: uv run --script scripts/plot-kernel-progression.py --experiment mage-003
The experiment selects the figure namespace under docs/assets; the stage values
are documented here and checked against the committed evidence namespaces.
Writes docs/assets/figures/<experiment>/kernel-progression.svg, its mobile
variant, and a downloadable PNG. The Pages build needs only the committed SVG.

The six stage values are the measurements quoted in the field note. Five of them
are read back from a committed evidence namespace and must match it: mage-001
(343.99 / 18.64 us), mage-002 (141.70 us) and mage-003 (80.00 / 10.05 us). The
seventh figure -- LayerNorm after the first rewrite, 11.03 us -- is the warp
kernel as measured in the mage-003 capture session; the mage-002 namespace was
captured in an earlier session and records 11.09 us for the same kernel, which
is inside the run-to-run spread this harness can resolve.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "mage-001"
BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
RUST = "#c9b2ff"
# Alpha ramp over one hue: the earliest arrangement is the faintest bar.
ALPHA = (.42, .68, 1.0)
TITLE = "How the two Rust kernels reached their measured kernel time"
SUBTITLE = ("GPU kernel time per stage, from separate Nsight Systems captures of 100 iterations on the same FP32 "
            "shapes. The label at each stage is the change that was made; the factor is that step's speed-up.")
FOOTER = ("Not adopted, all single-run development measurements with the metric named: matrix multiplication with "
          "32 × 32 tiles and a 2 × 4 register tile 210.51 us and an 8 × 4 register tile on 128-row blocks 147.74 us, "
          "both CUDA-event spans; LayerNorm with 128-thread blocks 14.4, a row staged in shared memory 14.3, eight named "
          "quad registers 13.4-14.0, and four warps per row 13.3-13.7, all event spans, and a row in a [F32x4; 8] array "
          "22.12 us of kernel time. RTX 4090, WSL2, unlocked clocks. Lower is better.")
# (stage, kernel microseconds, evidence namespace to check against, change made at this stage)
OPS = (
    ("matmul", "Matrix multiplication", (
        ("mage-001", 343.99, BASELINE,
         "16 × 16 tiles, one output per thread · two shared reads per multiply-add"),
        ("PR #42", 141.70, "mage-002",
         "4 × 4 outputs per thread, 64 × 64 block tiles · 128-bit global loads · K step 32 · 0.5 shared reads per multiply-add"),
        ("PR #43", 80.00, "mage-003",
         "B tile read as 128-bit quads · A tile transposed k-major, row stride 68 · K step 64 · 0.125 shared reads per multiply-add"),
    )),
    ("layernorm", "LayerNorm", (
        ("mage-001", 18.64, BASELINE,
         "one 256-thread block per row · three scalar passes · two shared tree reductions · about 18 barriers"),
        ("PR #42", 11.03, None,
         "one warp per row · 128-bit quads · shuffle reductions replace the barriers"),
        ("PR #44", 10.05, "mage-003",
         "two warps per row, half a row each · one shared exchange behind a single barrier · 1024 blocks of 256 threads"),
    )),
)


def fold(text, fontsize, width, weight="normal"):
    """Break a caption into lines that fit the label column, so mobile stays readable."""
    prop = FontProperties(family="DejaVu Sans", size=fontsize, weight=weight)
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        extent = TextPath((0, 0), candidate, prop=prop).get_extents().width * 100 / 72
        if extent <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)


def kernel_times(namespace):
    """Rust GPU kernel time per operation from one committed evidence namespace."""
    data = json.loads((ROOT / "docs/assets/results" / namespace / "comparison-profiles.json").read_text())
    times = {}
    for capture in data["captures"]:
        if capture["language"] != "rust":
            continue
        durations = [metric["duration_us"] for metric in capture["kernel_metrics"]]
        assert len(durations) == capture["launches"] and capture["status"] == "complete"
        times[capture["operation"]] = sum(durations) / capture["requested_iterations"]
    return times


def checked(measured):
    """Return the documented stages with every value that has an evidence namespace verified."""
    stages = []
    for op, label, steps in OPS:
        rows = []
        for name, value, namespace, change in steps:
            if namespace is not None:
                recorded = measured[namespace][op]
                assert abs(recorded - value) < 0.05, (op, name, recorded, value)
            rows.append((name, value, change))
        stages.append((op, label, rows))
    return stages


def plot(stages, mobile):
    width, height = (3.65, 9.4) if mobile else (12.2, 8.4)
    fig, axes = plt.subplots(len(stages), 1, figsize=(width, height), dpi=100)
    fig.subplots_adjust(left=.50 if mobile else .315, right=.98, top=.855 if mobile else .885,
                        bottom=.13 if mobile else .155, hspace=.62 if mobile else .55)
    fig.patch.set_facecolor(BG)
    label_pt, note_pt = (6.6, 6.2) if mobile else (9.5, 7.6)
    label_width = width * (0.50 if mobile else 0.315) * 100 - 18
    for ax, (op, label, rows) in zip(axes, stages):
        ax.set_facecolor(BG)
        for index, (name, value, _) in enumerate(rows):
            ax.barh(index, value, height=.34, color=RUST, alpha=ALPHA[index], zorder=3)
            ax.annotate(f"{value:.2f}", (value, index), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=label_pt - .5)
            ax.text(-.02, index - .20, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color=INK, fontsize=label_pt)
            ax.text(-.02, index - .02, fold(rows[index][2], note_pt, label_width),
                    transform=ax.get_yaxis_transform(), ha="right", va="top", color=MUTED, fontsize=note_pt,
                    linespacing=1.45)
            if index:
                previous = rows[index - 1][1]
                ax.plot([previous, previous, value, value], [index - 1, index - .5, index - .5, index],
                        color=MUTED, linewidth=.9, linestyle=(0, (3, 3)), zorder=2)
                ax.text((previous + value) / 2, index - .5, f"{previous / value:.2f}×", ha="center", va="center",
                        color=INK, fontsize=note_pt, zorder=4,
                        bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.6})
        span = max(value for _, value, _ in rows)
        limit = span * (1.30 if mobile else 1.22)
        ax.set_xlim(0, limit)
        ax.set_ylim(2.66, -.62)
        ax.set_yticks([])
        ax.set_title(label, loc="left", color=INK, fontsize=label_pt + 3, pad=8, weight="medium")
        ticks = MaxNLocator(nbins=3 if mobile else 5).tick_values(0, limit)
        ax.set_xticks([tick for tick in ticks if 0 <= tick <= limit])
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.6 if mobile else 9)
        ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    caption = width * 100 - 24
    fig.text(.01, .985, fold(TITLE, 11 if mobile else 16, caption, "bold"), color=INK,
             fontsize=11 if mobile else 16, weight="bold", va="top")
    fig.text(.01, .945 if mobile else .935, fold(SUBTITLE, 7.6 if mobile else 9.6, caption), color=MUTED,
             fontsize=7.6 if mobile else 9.6, va="top")
    fig.text(.01, .012, fold(FOOTER, 6.8 if mobile else 8.4, caption), color=MUTED,
             fontsize=6.8 if mobile else 8.4, va="bottom")
    name = f"kernel-progression{'-mobile' if mobile else ''}"
    target = ROOT / "docs/assets/figures" / EXPERIMENT / f"{name}.svg"
    fig.savefig(target, metadata={"Date": None, "Description":
                f"RTX 4090; same FP32 inputs. GPU kernel time per stage for the {EXPERIMENT} kernel revisions. "
                "Stages without a retained capture are named in comparison-profiles.json namespaces."})
    target.write_bytes(b"\n".join(line.rstrip() for line in target.read_bytes().splitlines()) + b"\n")
    if not mobile:
        fig.savefig(ROOT / "docs/assets/figures" / EXPERIMENT / f"{name}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=os.environ.get("MAGE_EXPERIMENT", "mage-003"),
                        help="namespace under docs/assets/figures for the written files")
    args = parser.parse_args()
    EXPERIMENT = args.experiment
    # Embed glyph outlines so downloads render identically without local fonts.
    # The page provides descriptive alt text and an accessible numeric table.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": f"{EXPERIMENT}-kernel-progression"})
    (ROOT / "docs/assets/figures" / EXPERIMENT).mkdir(parents=True, exist_ok=True)
    measured = {}
    for _, _, steps in OPS:
        for _, _, namespace, _ in steps:
            if namespace is not None and namespace not in measured:
                measured[namespace] = kernel_times(namespace)
    stages = checked(measured)
    for mobile in (False, True):
        plot(stages, mobile)
    for op, label, rows in stages:
        print(f"{op:10} " + "  ".join(f"{name} {value:7.2f}" for name, value, _ in rows))
