"""Schematic: one layer-norm row divided between warps, before and after.

Not a measurement — a drawing of the work split, taken from the two kernels in
docs/experiments/mage-003.md, with the 128-bit lane loads, the shared-memory
exchange and what each version asks of an SM's 1536 threads annotated. It exists
because the step that moved this kernel was a change of split rather than of
arithmetic: the same 768-float row, read by one warp and then by two.

Run: .venv/bin/python scripts/plot-layernorm-row-split.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/figures/mage-003"

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
WARP0, WARP1, SHARED, WARM = "#91dbba", "#c9b2ff", "#93caff", "#e0a08a"

BAR_X0, BAR_X1 = 22.0, 146.0  # the row strip
BAR_Y, BAR_H = 70.0, 8.0
BOX_X0, BOX_X1 = 28.0, 140.0  # the band under the row: what crosses between warps
BOX_Y0, BOX_Y1 = 36.0, 52.0
SLOT_X, SLOT_W, SLOT_GAP = 44.0, 10.0, 2.0
SCALE = 1.0  # mobile shrinks every label so both variants fit the same 150-unit panels


def text(ax, x, y, s, color=MUTED, size=8.2, ha="left", weight="normal"):
    ax.text(x, y, s, color=color, fontsize=size * SCALE, ha=ha, va="top", weight=weight, zorder=6)


def lane_strip(ax, x0, x1, lanes, face, mark=7):
    """One row drawn as `lanes` lane slices; the marked lane is the one the labels describe."""
    width = (x1 - x0) / lanes
    for index in range(lanes):
        lit = index == mark
        ax.add_patch(Rectangle((x0 + index * width, BAR_Y), width - .35, BAR_H,
                               facecolor=face, edgecolor=INK if lit else BG,
                               linewidth=1.1 if lit else .45, zorder=3))


def sm_slots(ax, filled, asked):
    """One SM's 1536-thread budget as six 256-thread slots, plus what the grid asks beyond it."""
    for index in range(6):
        ax.add_patch(Rectangle((SLOT_X + index * (SLOT_W + SLOT_GAP), 14), SLOT_W, 8,
                               facecolor=WARM if index < filled else "none",
                               edgecolor=WARM if index < filled else RULE,
                               linewidth=.9, zorder=3))
    for index in range(asked):
        ax.add_patch(Rectangle((SLOT_X + (6 + index) * (SLOT_W + SLOT_GAP), 14), SLOT_W, 8,
                               facecolor="none", edgecolor=WARM, linewidth=.9,
                               linestyle=(0, (2.5, 2.5)), zorder=3))
    if asked:
        text(ax, 127, 30, "asked, not resident", WARM, 8.2, ha="center")


def exchange(ax, solid, lines):
    edge = SHARED if solid else RULE
    ax.add_patch(Rectangle((BOX_X0, BOX_Y0), BOX_X1 - BOX_X0, BOX_Y1 - BOX_Y0, facecolor="none",
                           edgecolor=edge, linewidth=1.1 if solid else .9,
                           linestyle="-" if solid else (0, (3, 3)), zorder=2))
    for offset, line in zip((10.6, 4.8), lines):
        text(ax, (BOX_X0 + BOX_X1) / 2, BOX_Y0 + offset, line, edge, 8.4, ha="center")


def panel_frame(ax, title, subtitle):
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(BG)
    text(ax, 2, 98, title, INK, 11.5, weight="bold")
    text(ax, 2, 91, subtitle, MUTED, 8.6)
    text(ax, BAR_X0, 85, r"row $i$: 768 floats $= 3072$ B", INK, 8.6)
    text(ax, BAR_X1, 85, "one layer-norm row", MUTED, 8.6, ha="right")


def before_panel(ax):
    panel_frame(ax, "before: one warp owns the whole row",
                "layer_norm_warp · 4096 rows in 512 blocks of 256 threads")
    lane_strip(ax, BAR_X0, BAR_X1, 32, WARP0)
    text(ax, (BAR_X0 + BAR_X1) / 2, 68.5, "32 lanes · 24 floats each", WARP0, 8.4, ha="center")
    text(ax, BAR_X0, 62.5, "each lane: 24 floats = 6 × 128-bit quads")
    text(ax, BAR_X0, 56.5, "reduction: shuffle_down 16, 8, 4, 2, 1", WARP0, 8.4)
    exchange(ax, False, ("no exchange between warps: 0 B of shared memory",
                         "the warp's partial sums never leave their lanes"))
    text(ax, BAR_X0, 34.5, "block = 8 warps = 8 rows · 512 blocks", INK, 8.4)
    text(ax, 2, 30, "one SM: 1536 threads = six 256-thread slots")
    sm_slots(ax, filled=4, asked=0)
    text(ax, 2, 9.5, "131072 threads over 128 SMs: at most 1024 resident per SM", INK, 8.6, weight="bold")
    text(ax, 2, 4, "the grid is short of the threads that keep loads in flight")


def after_panel(ax):
    panel_frame(ax, "after: two warps split the row",
                "layer_norm_pair · 4096 rows in 1024 blocks of 256 threads")
    middle = (BAR_X0 + BAR_X1) / 2
    lane_strip(ax, BAR_X0, middle, 32, WARP0)
    lane_strip(ax, middle, BAR_X1, 32, WARP1)
    text(ax, (BAR_X0 + middle) / 2, 68.5, "warp 0 · 12 floats per lane", WARP0, 8.4, ha="center")
    text(ax, (middle + BAR_X1) / 2, 68.5, "warp 1 · 12 floats per lane", WARP1, 8.4, ha="center")
    text(ax, BAR_X0, 62.5, "each lane: 12 floats = 3 × 128-bit quads")
    for x_from, color in ((middle - 24, WARP0), (middle + 24, WARP1)):
        ax.add_patch(FancyArrowPatch((x_from, 57.5), (x_from, BOX_Y1 + .5), arrowstyle="-|>",
                                     mutation_scale=9, color=color, linewidth=1.3, zorder=5))
    exchange(ax, True, (r"each half sums $x$ and $x^2$ in one pass",
                        "the two partial sums meet in 64 B of shared, once"))
    text(ax, BAR_X0, 34.5, "block = 8 warps = 4 rows · 1024 blocks", INK, 8.4)
    text(ax, 2, 30, "one SM: 1536 threads = six 256-thread slots")
    sm_slots(ax, filled=6, asked=2)
    text(ax, 2, 9.5, "1024 blocks: twice the threads asked of the same 128 SMs", INK, 8.6, weight="bold")
    text(ax, 2, 4, "the split reaches the resident-thread limit; more warps do not")


def render(wide):
    """Draw both panels at one size; the mobile variant stacks them and shrinks the labels."""
    global SCALE
    SCALE = 1.0 if wide else .8
    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
        fig.subplots_adjust(left=.005, right=.995, top=.795, bottom=.01, wspace=.05)
        fig.text(.005, .975, "One row, one warp, then two", color=INK, fontsize=13, weight="bold", va="top")
        fig.text(.005, .925, "Schematic, not a measurement: the same 768-float row and the same 128-bit loads, "
                             "read by one warp and then by two.", color=MUTED, fontsize=9, va="top")
        name = "layernorm-row-split"
    else:
        fig, axes = plt.subplots(2, 1, figsize=(4.2, 6.6))
        fig.subplots_adjust(left=.02, right=.98, top=.895, bottom=.02, hspace=.12)
        fig.text(.02, .975, "One row, one warp, then two", color=INK, fontsize=10.5, weight="bold", va="top")
        fig.text(.02, .937, "Schematic of the layer-norm row split, before and after", color=MUTED,
                 fontsize=8, va="top")
        name = "layernorm-row-split-mobile"
    before_panel(axes[0])
    after_panel(axes[1])
    fig.patch.set_facecolor(BG)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Description":
                "Schematic of one layer-norm row split between one warp and two warps; not a "
                "measurement. See docs/experiments/mage-003.md."}
    fig.savefig(OUT / f"{name}.svg", metadata=metadata)
    svg = OUT / f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    if wide:
        fig.savefig(OUT / f"{name}.png", dpi=200, metadata=metadata)
    plt.close(fig)


def main():
    for wide in (True, False):
        render(wide)
    print("wrote", OUT / "layernorm-row-split.svg")


if __name__ == "__main__":
    # Embed glyph outlines so downloads render identically without local fonts.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "mage-003-layernorm-row-split"})
    main()
