"""Two schematics for the male-CNS mapping note.

`fly-cns-csr.svg` — what the connectome has to become to be computable: a node
table and a CSR edge list, sorted and blocked by cell type, with the two kernels
that read it annotated.

`fly-cns-loop.svg` — where those numbers go: the workstation that computes them,
the artifact it produces, the on-robot CUDA contract that loads it, and the
planner cost term and safety gate that decide what happens to it.

Numbers are the ones quoted in the note; nothing here is drawn from a
computation. Schematic, not a measurement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
GREEN, LAVENDER, BLUE, WARM = "#91dbba", "#c9b2ff", "#93caff", "#e0a08a"

OUT = Path("docs/assets/figures/fly-cns")


def frame(fig, width, height, title, subtitle):
    fig.patch.set_facecolor(BG)
    fig.text(0.012, 0.955, title, color=INK, fontsize=11.5, weight="bold", va="top")
    fig.text(0.012, 0.915, subtitle, color=MUTED, fontsize=8.6, va="top")
    fig.text(0.988, 0.02, "schematic, not a measurement", color=RULE, fontsize=7.6,
             ha="right", va="bottom")
    fig.set_size_inches(width, height)


def panel(ax, title, note):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.955, title, transform=ax.transAxes, color=INK, fontsize=9.6,
            weight="bold", va="top")
    ax.text(0.02, 0.885, note, transform=ax.transAxes, color=MUTED, fontsize=8.0, va="top")


def block(ax, x, y, w, h, label, sub, color, alpha=0.16, fontsize=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha))
    ax.text(x + w / 2, y + h / 2 + (0.030 if sub else 0.0), label, color=INK, fontsize=fontsize,
            ha="center", va="center", weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.032, sub, color=MUTED, fontsize=fontsize - 0.8,
                ha="center", va="center")


def arrow(ax, start, end, color=RULE, style="-|>", radius=0.02, lw=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={radius}"))


def layouts():
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.6), gridspec_kw={"hspace": 0.28})
    frame(fig, 11.0, 5.6,
          "The male CNS as the arrays a kernel reads",
          "166,700 neurons, 11,710 types, 45.6M pre-synaptic sites, 311.8M post-synaptic sites"
          " — and 929,735 type-to-type edges, which is the whole graph a GPU has to hold")

    # --- the node table, blocked by cell type -------------------------------
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "Rows: one per neuron, sorted by type",
          "body id, type, superclass, hemilineage, side, dimorphism flag, predicted neurotransmitter")

    blocks = [
        ("isomorphic", "8,069 types", GREEN),
        ("dimorphic", "138", LAVENDER),
        ("male-specific", "289", WARM),
        ("female-specific", "71", BLUE),
    ]
    widths = [0.78, 0.075, 0.105, 0.04]
    x = 0.035
    for (label, sub, color), w in zip(blocks, widths):
        block(ax, x, 0.30, w, 0.36, label, sub, color, fontsize=7.6 if w < 0.1 else 8.2)
        x += w + 0.012
    ax.text(0.035, 0.19, "type counts, Berg et al. 2026 (Cell); the four classes are the entire"
                         " cross-sex comparison", color=MUTED, fontsize=7.4)
    ax.text(0.965, 0.19, "one row group per type  →  every per-type statistic is a segmented reduce",
            color=RULE, fontsize=7.4, ha="right")

    # --- the CSR edge list ---------------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "Edges: CSR, blocked the same way",
          "rowptr[i]..rowptr[i+1] indexes indices[] and weights[] for neuron i — the layout the"
          " neighbour kernel already reads")

    # rowptr strip
    ax.add_patch(Rectangle((0.035, 0.62), 0.93, 0.10, facecolor=BLUE, alpha=0.16,
                           edgecolor=BLUE, linewidth=1.0))
    for tick in (0.22, 0.44, 0.66, 0.84):
        ax.plot([0.035 + 0.93 * (tick - 0.035) / 0.93] * 2, [0.62, 0.72], color=RULE, lw=0.8)
    ax.text(0.05, 0.67, "rowptr[]", color=INK, fontsize=8.0, va="center", weight="bold")
    ax.text(0.945, 0.67, "166,701 offsets, u64 → 1.3 MB", color=MUTED, fontsize=7.6,
            va="center", ha="right")

    # indices / weights strips
    ax.add_patch(Rectangle((0.035, 0.40), 0.93, 0.12, facecolor=GREEN, alpha=0.16,
                           edgecolor=GREEN, linewidth=1.0))
    ax.text(0.05, 0.46, "indices[]  neighbour body id", color=INK, fontsize=8.0, va="center")
    ax.text(0.945, 0.46, "25,563,426 neuron-level edges", color=MUTED, fontsize=7.6,
            va="center", ha="right")

    ax.add_patch(Rectangle((0.035, 0.24), 0.93, 0.12, facecolor=LAVENDER, alpha=0.16,
                           edgecolor=LAVENDER, linewidth=1.0))
    ax.text(0.05, 0.30, "weights[]  synapse count per edge", color=INK, fontsize=8.0, va="center")
    ax.text(0.945, 0.30, "6,237,402 edges at weight ≥ 5", color=MUTED, fontsize=7.6,
            va="center", ha="right")

    # kernel annotations
    ax.text(0.035, 0.115, "reads:  degree, sign balance, per-type totals   "
                          "— one pass over indices[]",
            color=GREEN, fontsize=7.8)
    ax.text(0.505, 0.115, "two-hop reach = sparse × sparse,   "
                          "929,735 × 929,735 at type level",
            color=LAVENDER, fontsize=7.8)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fly-cns-csr.svg", facecolor=BG)
    fig.savefig(OUT / "fly-cns-csr.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / "fly-cns-csr.svg")


def loop():
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    frame(fig, 11.0, 4.0,
          "Where the numbers go",
          "the dataset becomes an array, the array becomes an artifact, and the artifact becomes"
          " a cost term — never an authority")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(RULE)

    y = 0.52
    h = 0.26
    block(ax, 0.02, y, 0.155, h, "male-cns:v1.0", "166,700 neurons\n13.1 GB of synapses", BLUE)
    block(ax, 0.205, y, 0.165, h, "workstation GPU", "CSR statistics,\nnull tests, two-hop", GREEN)
    block(ax, 0.40, y, 0.155, h, "artifact", "type-level prior\n+ risk table", LAVENDER)
    block(ax, 0.585, y, 0.155, h, "leash-cuda", "fatbin, SM 8.7\nJetson Orin NX", WARM)
    block(ax, 0.77, y, 0.155, h, "qualia planner", "plan_path,\nbelief_risk cost", BLUE)

    for x0, x1 in ((0.175, 0.205), (0.370, 0.400), (0.555, 0.585), (0.740, 0.770)):
        arrow(ax, (x0, y + h / 2), (x1, y + h / 2), color=RULE)

    # the advisory boundary
    ax.plot([0.02, 0.98], [0.40, 0.40], color=WARM, lw=1.0, linestyle=(0, (4, 3)))
    ax.text(0.02, 0.335, "advisory only: compute results cannot authorize or refresh motor output",
            color=WARM, fontsize=8.0)
    ax.text(0.98, 0.335, "leash keeps the gates — collision, deadman, stop, E-stop",
            color=WARM, fontsize=8.0, ha="right")

    ax.text(0.02, 0.86, "dataset", color=MUTED, fontsize=7.6)
    ax.text(0.205, 0.86, "compute", color=MUTED, fontsize=7.6)
    ax.text(0.40, 0.86, "compress", color=MUTED, fontsize=7.6)
    ax.text(0.585, 0.86, "load", color=MUTED, fontsize=7.6)
    ax.text(0.77, 0.86, "cost", color=MUTED, fontsize=7.6)

    ax.text(0.02, 0.16, "belief_risk { uncertainty_weight, semantic_novelty }  →  path_cost += "
                        "(uncertainty × 12 + novelty × 4) × grid resolution",
            color=MUTED, fontsize=7.8)
    ax.text(0.02, 0.07, "the planner today is grid-only (astar, uniform cost); a connectome enters"
                        " as a cost prior, not as the planner",
            color=RULE, fontsize=7.6)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fly-cns-loop.svg", facecolor=BG)
    fig.savefig(OUT / "fly-cns-loop.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / "fly-cns-loop.svg")


if __name__ == "__main__":
    layouts()
    loop()
