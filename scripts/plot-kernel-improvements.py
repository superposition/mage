# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Render the rewritten Rust kernels against the baselines and their own mage-001 values.

Run: uv run --script scripts/plot-kernel-improvements.py --experiment mage-002
The experiment selects the result and figure namespaces under docs/assets; the
baseline selects the earlier committed kernel times used for the "before" bars.
Writes docs/assets/figures/<experiment>/kernel-improvements.svg, its mobile
variant, and a downloadable PNG. The Pages build needs only the committed SVG.
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

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mage-002"
BASELINE = "mage-001"
OPS = (("matmul", "Matrix multiplication"), ("layernorm", "LayerNorm"))
SERIES = ("#93caff", "#91dbba", "#a0a9b9", "#c9b2ff")
BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
TITLE = "Rewriting the Rust kernels: GPU kernel time before and after"
SUBTITLE = ("Microseconds per operation. Separate Nsight Systems captures of 100 iterations per measurement, "
            "same FP32 inputs and shapes.")
FOOTER = ("RTX 4090, WSL2, unlocked clocks. Lower is better. Each panel has its own scale. "
          "PyTorch, Triton, and the {experiment} Rust bar share one capture session; "
          "the {baseline} Rust bar is the earlier one.")


def fold(text, fontsize, width, weight="normal"):
    """Break a caption into lines that fit the canvas, so the mobile variant stays readable."""
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
    """GPU kernel time per operation and implementation from one evidence namespace.

    Sum of captured kernel durations divided by the requested iterations, so a
    multi-kernel operation is compared as the complete operation.
    """
    data = json.loads((ROOT / "docs/assets/results" / namespace / "comparison-profiles.json").read_text())
    times = {}
    for capture in data["captures"]:
        durations = [metric["duration_us"] for metric in capture["kernel_metrics"]]
        assert len(durations) == capture["launches"] and capture["status"] == "complete"
        times[capture["operation"], capture["language"]] = sum(durations) / capture["requested_iterations"]
    return times


def plot(times, mobile):
    # Two panels, four bars each: the shared baselines, the earlier Rust kernel,
    # and the rewritten one. Each panel keeps its own zero-based scale.
    width, height = (3.65, 6.2) if mobile else (13.0, 6.0)
    if mobile:
        fig, axes = plt.subplots(len(OPS), 1, figsize=(width, height), dpi=100)
        fig.subplots_adjust(left=.325, right=.975, top=.825, bottom=.135, hspace=.62)
    else:
        fig, axes = plt.subplots(1, len(OPS), figsize=(width, height), dpi=100)
        fig.subplots_adjust(left=.135, right=.985, top=.70, bottom=.30, wspace=.34)
    fig.patch.set_facecolor(BG)
    caption_width = width * 100 - 24
    names = ["PyTorch", "Triton", f"Rust, {BASELINE}", f"Rust, {EXPERIMENT}"]
    for ax, (op, label) in zip(axes, OPS):
        before = times[BASELINE][op, "rust"]
        values = [times[EXPERIMENT][op, language] for language in ("python", "triton")]
        values += [before, times[EXPERIMENT][op, "rust"]]
        ax.barh(range(4), values, height=.55, color=SERIES, zorder=3)
        for index, value in enumerate(values):
            ax.text(value * 1.03, index, f"{value:.1f}", va="center", ha="left", color=INK,
                    fontsize=11.5 if not mobile else 10)
        ax.set_xlim(0, max(values) * (1.34 if mobile else 1.28))
        ax.set_ylim(3.6, -0.6)
        ax.set_yticks(range(4), names, color=MUTED, fontsize=10 if not mobile else 8.5)
        ax.set_xticks([])
        ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(BG)
        ratio = before / values[3]
        change = f"{before:.1f} → {values[3]:.1f} µs ({ratio:.2f}× faster)"
        if mobile:
            ax.set_title(f"{label}\n{before:.1f} → {values[3]:.1f} µs · {ratio:.2f}×",
                         loc="left", color=INK, fontsize=8.5, pad=5)
        else:
            ax.set_title(label, loc="left", color=INK, fontsize=14, pad=9)
            ax.text(0.0, -0.30, f"Rust: {change}", transform=ax.transAxes, color=INK,
                    fontsize=11, va="top")
    fig.text(.01, .985 if mobile else .965, fold(TITLE, 11 if mobile else 16, caption_width, "bold"),
             color=INK, fontsize=11 if mobile else 16, weight="bold", va="top")
    fig.text(.01, .935 if mobile else .895, fold(SUBTITLE, 8.2 if mobile else 10, caption_width),
             color=MUTED, fontsize=8.2 if mobile else 10, va="top")
    fig.text(.01, .012 if mobile else .035,
             fold(FOOTER.format(experiment=EXPERIMENT, baseline=BASELINE), 7.4 if mobile else 9,
                  caption_width),
             color=MUTED, fontsize=7.4 if mobile else 9, va="bottom")
    name = f"kernel-improvements{'-mobile' if mobile else ''}"
    target = ROOT / "docs/assets/figures" / EXPERIMENT / f"{name}.svg"
    fig.savefig(target, metadata={"Date": None, "Description":
                f"RTX 4090; same FP32 inputs. GPU kernel time per operation. Rust bars: {BASELINE} "
                f"committed values versus {EXPERIMENT}. See comparison-profiles.json for the captures."})
    target.write_bytes(b"\n".join(line.rstrip() for line in target.read_bytes().splitlines()) + b"\n")
    if not mobile:
        fig.savefig(ROOT / "docs/assets/figures" / EXPERIMENT / f"{name}.png", dpi=200)
    plt.close(fig)
    return {op: (times[BASELINE][op, "rust"], times[EXPERIMENT][op, "rust"]) for op, _ in OPS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=os.environ.get("MAGE_EXPERIMENT", "mage-002"),
                        help="namespace under docs/assets/results and docs/assets/figures")
    parser.add_argument("--baseline", default="mage-001", help="earlier experiment used for the before bars")
    args = parser.parse_args()
    EXPERIMENT, BASELINE = args.experiment, args.baseline
    # Embed glyph outlines so downloads render identically without local fonts.
    # The page provides descriptive alt text and an accessible numeric table.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": f"{EXPERIMENT}-kernel-improvements"})
    (ROOT / "docs/assets/figures" / EXPERIMENT).mkdir(parents=True, exist_ok=True)
    measured = {namespace: kernel_times(namespace) for namespace in (EXPERIMENT, BASELINE)}
    for mobile in (False, True):
        changed = plot(measured, mobile)
    for op, (before, after) in changed.items():
        print(f"{op:10} {before:7.2f} -> {after:7.2f} us")
