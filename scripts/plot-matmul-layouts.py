"""Two libraries, the same matrices, different memory.

Draws how cuTile Rust and cuda-oxide place the same C = A B problem: one tile per
program on the left, one register tile per thread on the right, with the shared
memory layouts and load widths annotated. Schematic, taken from both kernels'
code, not a measurement.

Run: uv run --script scripts/plot-matmul-layouts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/figures/mage-004"

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
CUTILE, OXIDE, SHARED = "#c9b2ff", "#91dbba", "#93caff"


def cell_grid(ax, x, y, cols, rows, size, face, edge=RULE, lw=.6, z=3):
    for c in range(cols):
        for r in range(rows):
            ax.add_patch(Rectangle((x + c * size, y - r * size), size - .6, size - .6,
                                   facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def label(ax, x, y, text, color=MUTED, size=8.2, ha="left", weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va="top", weight=weight)


def panel_title(ax, title, subtitle, color):
    ax.text(1, 98, title, color=color, fontsize=12, weight="bold", va="top")
    ax.text(1, 90.5, subtitle, color=MUTED, fontsize=8.6, va="top")


def cutile_panel(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")
    panel_title(ax, "cuTile Rust: one tile per program", "the compiler owns the threads and the layout", CUTILE)

    # the output, partitioned into 32 x 128 tiles
    ax.add_patch(Rectangle((8, 22), 62, 62, facecolor="none", edgecolor=RULE, linewidth=1, zorder=2))
    for c in range(4):
        for r in range(4):
            lit = (c, r) == (1, 1)
            ax.add_patch(Rectangle((8 + c * 15.5, 22 + r * 15.5), 15.5 - .7, 15.5 - .7,
                                   facecolor=CUTILE if lit else "none", edgecolor=RULE,
                                   linewidth=.8, alpha=.95 if lit else 1, zorder=3))
    label(ax, 8, 88, r"$C = A B$", color=INK, size=11)
    label(ax, 8, 84, r"$C: 1024 \times 1024$ in $32 \times 128$ tiles", size=8)
    label(ax, 8, 19, r"one program per lit tile", size=8, color=CUTILE)

    # what it reads per K step
    ax.add_patch(FancyArrowPatch((74, 59), (86, 59), arrowstyle="-|>", mutation_scale=10,
                                 color=CUTILE, linewidth=1.3, zorder=4))
    label(ax, 86, 66, r"$A$ tile $32\times32$", size=8, color=INK)
    label(ax, 86, 60, r"$B$ tile $32\times32$", size=8, color=INK)
    label(ax, 86, 54, r"$32$ steps of $K$", size=8)
    label(ax, 86, 47, "widened to 128-bit loads", size=8, color=MUTED)
    label(ax, 86, 41, "tile shape is fixed at", size=8, color=MUTED)
    label(ax, 86, 35.5, "compile time, so it is part", size=8, color=MUTED)
    label(ax, 86, 30, "of what gets compiled", size=8, color=MUTED)


def oxide_panel(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")
    panel_title(ax, "cuda-oxide: one register tile per thread", "the kernel author owns both", OXIDE)

    # the block tile and one thread's 4 x 4 registers
    ax.add_patch(Rectangle((8, 22), 62, 62, facecolor="none", edgecolor=RULE, linewidth=1, zorder=2))
    for c in range(16):
        for r in range(16):
            lit = 4 <= c < 8 and 4 <= r < 8
            ax.add_patch(Rectangle((8 + c * 3.875, 22 + r * 3.875), 3.875 - .35, 3.875 - .35,
                                   facecolor=OXIDE if lit else "none", edgecolor=RULE,
                                   linewidth=.4, zorder=3))
    label(ax, 8, 88, r"$C = A B$", color=INK, size=11)
    label(ax, 8, 84, r"$C: 64 \times 64$ per block, $16 \times 16$ threads", size=8)
    label(ax, 8, 19, "lit = one thread's 4 x 4 registers", size=8, color=OXIDE)

    # shared memory: A transposed so a thread's four rows are contiguous
    ax.add_patch(FancyArrowPatch((74, 59), (86, 59), arrowstyle="-|>", mutation_scale=10,
                                 color=OXIDE, linewidth=1.3, zorder=4))
    label(ax, 86, 66, r"shared $A^T$: $64 \times 68$", size=8, color=INK)
    label(ax, 86, 60, r"row stride 68, not 64", size=8, color=SHARED)
    label(ax, 86, 54, "one 128-bit read per", size=8, color=MUTED)
    label(ax, 86, 48, "row, four rows at once", size=8, color=MUTED)
    label(ax, 86, 41, r"shared $B$: $64 \times 64$", size=8, color=INK)
    label(ax, 86, 35.5, r"$0.125$ shared reads per", size=8, color=SHARED)
    label(ax, 86, 29.5, r"multiply-add", size=8, color=SHARED)
    label(ax, 86, 22, "stride 68 avoids bank conflicts", size=8, color=MUTED)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.02, wspace=.05)
    fig.text(.01, .965, "The same product, two ways of placing it in memory", color=INK,
             fontsize=13, weight="bold", va="top")
    fig.text(.01, .915, r"$C = A B$ at $1024^3$, FP32. The compiler's kernel and the hand-written one "
                        "move different amounts of data to reach the same answer.", color=MUTED,
             fontsize=9, va="top")
    cutile_panel(axes[0])
    oxide_panel(axes[1])
    for ax in axes:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Description":
                "Schematic of the matmul tile layouts in cuTile Rust and cuda-oxide; not a "
                "measurement. See docs/experiments/mage-004.md."}
    for name in ("matmul-layouts", "matmul-layouts-mobile"):
        if name.endswith("mobile"):
            fig.set_size_inches(4.2, 6.4)
            axes[0].set_position([.02, .52, .96, .36])
            axes[1].set_position([.02, .06, .96, .36])
        fig.savefig(OUT / f"{name}.svg", metadata=metadata)
        svg = OUT / f"{name}.svg"
        svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
        if not name.endswith("mobile"):
            fig.savefig(OUT / f"{name}.png", dpi=200, metadata=metadata)
    plt.close(fig)
    print("wrote", OUT / "matmul-layouts.svg")


if __name__ == "__main__":
    main()
