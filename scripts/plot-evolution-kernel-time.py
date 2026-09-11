# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Draw the matmul progression and the interaction that pointed at the next step.

Run: uv run --script scripts/plot-evolution-kernel-time.py [--check]

Left: the three Rust matrix-multiply kernels with the library references — the committed
kernel the loop started from, the configuration the loop retained, and the pipelined
kernel that superseded it. Right: the same configuration split into its two halves,
each row a paired session against the kernel as it stood before the pipeline landed.

Values are read back from the committed captures under
docs/assets/results/evolution-loop/kernel-time/ and must match them:

- four-arm-same-session.json (session A): registers 79.76, loop 76.34, Triton 84.51,
  cuBLAS 49.88 us per iteration.
- pipelined-baseline.json (session B, after PR #49): pipeline 68.76, loop 76.25,
  Triton 86.22, cuBLAS 50.50.
- attr-*.json and guarded-*.json (session A): shared(64) 80.31, guard-free(64) 80.29,
  guard-free(32) 82.43, guarded(64) 82.97, guarded(32) 76.29.

`--check` asserts every label lands inside the canvas and stops, because the figure
ships in a published field note and this environment cannot review it by eye.

Writes docs/assets/figures/mage-005/kernel-time-comparison.svg, its mobile variant, and
a downloadable PNG.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/assets/results/evolution-loop/kernel-time"
BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
RUST, BASE, LIB, GOOD = "#c9b2ff", "#6f7b8f", "#f0b46a", "#8fd6a8"
TITLE = "Three matmul kernels and the interaction behind the middle one"
SUBTITLE = ("GPU kernel time per iteration, Nsight Systems captures of 100 iterations at 1024^3 FP32, medians of three "
            "interleaved rounds. Left: the committed kernel the loop started from, the configuration it retained, and "
            "the pipelined kernel that superseded it, with Triton and cuBLAS from the same sessions. Right: the retained "
            "configuration split into its two halves, against the kernel as it stood before the pipeline landed.")
FOOTER = ("Sessions: the first two bars and the right panel from one session before PR #49; the third bar, Triton and "
          "cuBLAS from a session after it. The committed kernel reproduced its mage-003 record (80.00 us) in the first "
          "session. Triton's value swings 71.94-83.78 us between sessions, which is why it is captured in the same "
          "session rather than compared across them. Not shown: LayerNorm, where four configurations were measured and "
          "none beat the committed kernel.")


def evidence(name):
    return json.loads((EVIDENCE / name).read_text())


def arm_medians(capture):
    """Median per arm, from the capture's own rows."""
    values = {}
    for row in capture["rows"]:
        values.setdefault(row["arm"], []).append(row["per_iteration_us"])
    return {arm: sorted(samples)[len(samples) // 2] for arm, samples in values.items()}


def fold(fig, text, fontsize, max_fraction=0.99):
    """Wrap to the canvas width by measuring the rendered text, not estimating it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    available = fig.get_size_inches()[0] * fig.dpi * max_fraction
    probe = fig.text(0, 0, "", fontsize=fontsize)
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        probe.set_text(candidate)
        if line and probe.get_window_extent(renderer).width > available:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    probe.remove()
    return "\n".join(lines)


def report_layout(fig):
    """Every label must land inside the canvas; the check exists because this figure ships."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.get_size_inches() * fig.dpi
    bad = []
    for text in fig.findobj(mtext.Text):
        if not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width_px + 0.5 or box.y1 > height_px + 0.5:
            bad.append((text.get_text()[:40], round(box.x0), round(box.y0), round(box.x1), round(box.y1)))
    print("canvas %.0f x %.0f px" % (width_px, height_px))
    if bad:
        for entry in bad:
            print("  OUTSIDE", entry)
        raise SystemExit("%d label(s) outside the canvas" % len(bad))
    print("all labels inside the canvas:", sum(1 for t in fig.findobj(mtext.Text) if t.get_text().strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="mage-005")
    parser.add_argument("--check", action="store_true",
                        help="assert every label lands inside the canvas, then stop")
    args = parser.parse_args()

    before = arm_medians(evidence("four-arm-same-session.json"))
    after = arm_medians(evidence("pipelined-baseline.json"))
    interaction = [
        ("shared, 64 (the kernel then)", evidence("attr-k64_shared.json"), "retained"),
        ("guard-free, 64", evidence("attr-k64_exact.json"), "retained"),
        ("guard-free, 32", evidence("attr-k32_exact.json"), "retained"),
        ("guarded, 64", evidence("guarded-k64-vs-k32.json"), "retained"),
        ("guarded, 32 (the combination)", evidence("guarded-k64-vs-k32.json"), "compare"),
    ]
    interaction_medians = [arm_medians(capture)[arm] for _, capture, arm in interaction]

    progression = [
        ("committed, registers", before["committed"], BASE, "session A"),
        ("the loop's configuration", before["retained"], RUST, "session A"),
        ("committed, pipelined", after["committed"], GOOD, "session B"),
    ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.9), dpi=100,
                                      gridspec_kw={"width_ratios": [1, 1.10], "wspace": .30})
    fig.patch.set_facecolor(BG)
    for ax in (left, right):
        ax.set_facecolor(BG)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)
        ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
        ax.set_xlim(0, 100)

    for index, (label, value, colour, session) in enumerate(progression):
        left.barh(index, value, height=.42, color=colour, zorder=3)
        left.text(value - 2.0, index, f"{value:.2f}", ha="right", va="center", color=BG,
                  fontsize=10.5, zorder=5)
        left.text(value + 1.4, index, session, ha="left", va="center", color=MUTED, fontsize=8.6)
    left.axvline(before["triton"], color=LIB, linewidth=.9, linestyle=(0, (4, 3)), zorder=2)
    left.axvline(after["pytorch"], color=MUTED, linewidth=.9, linestyle=(0, (4, 3)), zorder=2)
    left.text(before["triton"] + .8, 2.42, f"Triton {after['triton']:.1f}", color=LIB, fontsize=8.6)
    left.text(after["pytorch"] + .8, 2.42, f"cuBLAS {after['pytorch']:.1f}", color=MUTED, fontsize=8.6)
    left.set_yticks(range(len(progression)), [label for label, _, _, _ in progression],
                    color=INK, fontsize=10.5)
    left.invert_yaxis()
    left.set_xlabel("us per iteration", color=MUTED, fontsize=9)
    left.set_title("The matmul, and what answered the loop", loc="left", color=INK, fontsize=12,
                   pad=10, weight="medium")

    reference = interaction_medians[0]
    for index, ((label, _, _), value) in enumerate(zip(interaction, interaction_medians)):
        highlight = label.startswith("guarded, 32")
        right.barh(index, value, height=.42, color=RUST if highlight else BASE, zorder=3)
        right.text(value - 2.0, index, f"{value:.2f}", ha="right", va="center", color=BG,
                   fontsize=10.5, zorder=5)
        right.text(value + 1.4, index, f"{value / reference:.3f}x", ha="left", va="center",
                   color=MUTED, fontsize=8.4)
    right.axvline(reference, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)), zorder=2)
    right.set_yticks(range(len(interaction)), [label for label, _, _ in interaction], color=INK,
                     fontsize=10.5)
    right.invert_yaxis()
    right.set_xlabel("us per iteration", color=MUTED, fontsize=9)
    right.set_title("The win was the combination (session A)", loc="left", color=INK, fontsize=12,
                    pad=10, weight="medium")

    fig.subplots_adjust(left=.235, right=.98, top=.755, bottom=.20)
    fig.text(.01, .975, fold(fig, TITLE, 17), color=INK, fontsize=17, weight="bold", va="top")
    fig.text(.01, .925, fold(fig, SUBTITLE, 9.4), color=MUTED, fontsize=9.4, va="top", linespacing=1.5)
    fig.text(.01, .012, fold(fig, FOOTER, 8.2), color=MUTED, fontsize=8.2, va="bottom", linespacing=1.5)

    if args.check:
        report_layout(fig)
        return

    target = ROOT / "docs/assets/figures" / args.experiment
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / "kernel-time-comparison.svg", facecolor=BG,
                metadata={"Date": None, "Description": "Kernel time per iteration"})
    fig.savefig(target / "kernel-time-comparison.png", dpi=200, facecolor=BG)
    fig.set_size_inches(6.4, 8.6)
    fig.subplots_adjust(left=.33)
    fig.savefig(target / "kernel-time-comparison-mobile.svg", facecolor=BG,
                metadata={"Date": None, "Description": "Kernel time per iteration"})
    print("wrote", target / "kernel-time-comparison.svg")


if __name__ == "__main__":
    main()
