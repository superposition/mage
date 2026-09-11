"""What is in flight while a K tile is multiplied, before and after.

Schematic, taken from the two kernels' code and from the resource request each
one reports. The kernel that entered this stage copies a K tile into shared
memory, synchronizes the block, and only then multiplies, so no copy is in
flight while the multiply-adds run. The pipeline keeps the same 33792 bytes of
shared memory as two buffers and issues the next tile's copy with cp.async, so
it runs underneath them. Not a measurement.

Run: .venv/bin/python scripts/plot-matmul-load-in-flight.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/figures/mage-006"

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
WARM, ACCENT, TILE_A, TILE_B = "#e0a08a", "#91dbba", "#c9b2ff", "#93caff"

STEP_0, PITCH, W = 22.0, 43.0, 20.0
COPY_Y, MMA_Y, BAR_H = 16.0, 4.0, 10.0


def label(ax, x, y, text, color=MUTED, size=8.2, ha="left", va="top", weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, weight=weight)


def tile_grid(ax, x, y, cols, rows, size, face):
    for c in range(cols):
        for r in range(rows):
            ax.add_patch(Rectangle((x + c * size, y - r * size), size - .8, size - .8,
                                   facecolor=face, edgecolor=RULE, linewidth=.5, zorder=3))


def bar(ax, x, y, width, face, text, text_color=BG):
    ax.add_patch(Rectangle((x, y), width, BAR_H, facecolor=face, edgecolor="none", zorder=3))
    label(ax, x + width / 2, y + BAR_H / 2, text, color=text_color, size=7.4,
          ha="center", va="center", weight="bold")


def timeline(ax, double):
    """Two K steps of the contraction as two rows of boxes."""
    label(ax, 2, 42, "shared: 33792 B,", color=INK, size=8)
    label(ax, 2, 36, "two buffers" if double else "one buffer", color=INK, size=8)
    label(ax, 2, 21, "copy", color=MUTED, size=8, va="center")
    label(ax, 2, 9, "multiply", color=MUTED, size=8, va="center")
    for i in range(2):
        x = STEP_0 + i * PITCH
        bar(ax, x, COPY_Y, W, ACCENT if double else WARM,
            "buffer %d" % (1 - i) if double else (r"tile $t$" if i == 0 else r"tile $t{+}1$"),
            INK if double else BG)
        if double:
            bar(ax, x, MMA_Y, W, MUTED, r"$\times\, t$" if i == 0 else r"$\times\, t{+}1$", BG)
        else:
            ax.add_patch(Rectangle((x + W + .8, MMA_Y), .9, BAR_H + 12, facecolor=RULE, zorder=4))
            bar(ax, x + W + 3, MMA_Y, W, MUTED, r"$\times\, t$" if i == 0 else r"$\times\, t{+}1$", BG)
            if i == 0:
                label(ax, x + W + 2.5, 34, "barrier", color=RULE, size=8, weight="bold")
    label(ax, 108, 9, "\u2026", color=MUTED, size=11, va="center")


def panel(ax, title, subtitle, note, note_color, double):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")
    label(ax, 2, 98, title, color=INK, size=11.5, weight="bold")
    label(ax, 2, 91, subtitle, color=MUTED, size=8.4)
    label(ax, 2, 85, note, color=note_color, size=8.6, weight="bold")

    label(ax, 2, 79, "per K step, one block reads", color=MUTED, size=8.4, weight="bold")
    tile_grid(ax, 2, 74, 8, 4, 5.0, TILE_A)
    label(ax, 46, 73, r"$A$ tile $64 \times 32$", color=INK, size=8.2)
    label(ax, 46, 67, "4-byte copies", color=TILE_A, size=8)
    label(ax, 2, 50, "the transpose scatters A's destination", color=TILE_A, size=8)
    tile_grid(ax, 74, 74, 4, 8, 4.0, TILE_B)
    label(ax, 74, 38, r"$B$ tile $32 \times 64$", color=INK, size=8.2)
    label(ax, 74, 32, "16-byte copies", color=TILE_B, size=8)

    timeline(ax, double)


def figure(width, height, title, subtitle, title_size=13, sub_size=9):
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.02, wspace=.06)
    fig.text(.01, .965, title, color=INK, fontsize=title_size, weight="bold", va="top")
    fig.text(.01, .915, subtitle, color=MUTED, fontsize=sub_size, va="top")
    panel(axes[0], "before: one buffer", "the copy, the barrier, then the multiply-adds",
          "nothing is in flight under the arithmetic", WARM, False)
    panel(axes[1], "after: two buffers", "the next tile's copy is issued with cp.async",
          "cp_async_wait_group(1) leaves one outstanding", ACCENT, True)
    for ax in axes:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    return fig, axes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Description":
                "Schematic of the K-tile buffers and the copies in flight in the matmul kernels "
                "before and after PR #49; not a measurement. See docs/experiments/mage-006.md."}

    fig, _ = figure(10.4, 3.9, "One K step's tile, and what is in flight while it is multiplied",
                    r"$64 \times 64$ block tile, $16 \times 16$ threads, $4 \times 4$ outputs per "
                    "thread, the same 33792 bytes of shared memory in both kernels.")
    fig.savefig(OUT / "matmul-load-in-flight.svg", metadata=metadata)
    svg = OUT / "matmul-load-in-flight.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    fig.savefig(OUT / "matmul-load-in-flight.png", dpi=200, metadata=metadata)
    plt.close(fig)

    # the stacked variant the entry references on narrow screens
    fig, _ = figure(4.2, 6.6, "One K step, and the copy in flight", "the same block tile, two ways "
                    "of loading it", title_size=10.5, sub_size=8)
    fig.axes[0].set_position([.03, .52, .94, .36])
    fig.axes[1].set_position([.03, .06, .94, .36])
    fig.savefig(OUT / "matmul-load-in-flight-mobile.svg", metadata=metadata)
    mobile = OUT / "matmul-load-in-flight-mobile.svg"
    mobile.write_bytes(b"\n".join(line.rstrip() for line in mobile.read_bytes().splitlines()) + b"\n")
    plt.close(fig)
    print("wrote", OUT / "matmul-load-in-flight.svg")


if __name__ == "__main__":
    main()
