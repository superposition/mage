"""Schematic: what one thread reads in the neighbor kernel, before and after.

Not a measurement — a drawing of the access pattern, taken from the two kernels'
code, with the load widths and the traffic they generate annotated. It exists
because the change is easier to see than to read: the same work, addressed
differently.

Run: uv run --script scripts/plot-neighbor-access.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/figures/mage-007"

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
ACCENT, WARM = "#91dbba", "#e0a08a"
FLOAT_W, FLOAT_H = 13.0, 11.0


def draw_rows(ax, x0, y0, rows, cols, label, color, quad_span=None):
    """Draw a few x[] rows as grids of floats, optionally highlighting a quad."""
    for r in range(rows):
        for c in range(cols):
            highlighted = quad_span is not None and quad_span[0] <= c < quad_span[1]
            ax.add_patch(Rectangle((x0 + c * FLOAT_W, y0 - r * FLOAT_H), FLOAT_W - 1.4, FLOAT_H - 1.4,
                                   facecolor=color if highlighted else RULE,
                                   edgecolor=INK if highlighted else "none", linewidth=.9, zorder=3))
        ax.text(x0 - 6, y0 - r * FLOAT_H + FLOAT_H / 2 - 1, label if r == 0 else "",
                color=MUTED, fontsize=8, ha="right", va="center")


def panel(ax, title, subtitle, load_bits, per_thread, edges_in_flight, quad_span, color):
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(2, 92, title, color=INK, fontsize=11.5, weight="bold", va="top")
    ax.text(2, 83, subtitle, color=MUTED, fontsize=8.6, va="top")

    # the x[] rows this thread reads through its edges
    ax.text(2, 66, "x rows named by the edges", color=MUTED, fontsize=8.4, va="top")
    draw_rows(ax, 46, 60, rows=3, cols=8, label="row", color=color, quad_span=quad_span)
    ax.text(46, 22, f"{per_thread} load{'s' if per_thread > 1 else ''} per thread per edge, "
                    f"{load_bits} bits each", color=color, fontsize=9, va="top", weight="bold")
    ax.text(46, 13, f"{edges_in_flight} edge{'s' if edges_in_flight > 1 else ''} in flight", color=MUTED,
            fontsize=8.6, va="top")

    # the arrow that stands for the gather
    for i in range(min(edges_in_flight, 2)):
        ax.add_patch(FancyArrowPatch((40, 62 - i * 14), (44.5, 58 - i * 14), arrowstyle="-|>",
                                     mutation_scale=9, color=color, linewidth=1.4, zorder=4))
    ax.text(2, 30, "weights, indices\nread per edge", color=MUTED, fontsize=8.4, va="top")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.02, wspace=.06)
    fig.text(.01, .965, "One thread's memory traffic in the neighbor kernel", color=INK,
             fontsize=13, weight="bold", va="top")
    fig.text(.01, .915, "Each output row is a sum over its edges; the question is how many bytes one "
                        "thread asks for at a time", color=MUTED, fontsize=9, va="top")
    panel(axes[0], "before", "one feature per thread, one edge per step", 32, 1, 1, None, WARM)
    panel(axes[1], "after", "four features per thread, two edges unrolled", 128, 1, 2, (2, 6), ACCENT)
    for ax in axes:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Description":
                "Schematic of the neighbor kernel's access pattern before and after the change; "
                "not a measurement. See docs/experiments/mage-007.md."}
    fig.savefig(OUT / "neighbor-access.svg", metadata=metadata)
    svg = OUT / "neighbor-access.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    fig.savefig(OUT / "neighbor-access.png", dpi=200, metadata=metadata)
    plt.close(fig)

    # the stacked variant the notes reference on narrow screens
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.4))
    fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.03, hspace=.35)
    fig.text(.02, .975, "One thread's memory traffic", color=INK, fontsize=10.5, weight="bold", va="top")
    fig.text(.02, .935, "in the neighbor kernel, before and after", color=MUTED, fontsize=8, va="top")
    panel(axes[0], "before", "one feature per thread, 32-bit loads", 32, 1, 1, None, WARM)
    panel(axes[1], "after", "four features per thread, 128-bit loads", 128, 1, 2, (2, 6), ACCENT)
    for ax in axes:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    fig.savefig(OUT / "neighbor-access-mobile.svg", metadata=metadata)
    mobile = OUT / "neighbor-access-mobile.svg"
    mobile.write_bytes(b"\n".join(line.rstrip() for line in mobile.read_bytes().splitlines()) + b"\n")
    plt.close(fig)
    print("wrote", OUT / "neighbor-access.svg")


if __name__ == "__main__":
    main()
