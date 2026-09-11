"""Draw what each kernel-lane change measured: before, after, and the references.

Reads docs/assets/results/mage-007/kernel-lane-fixes.json and writes the figure to
docs/assets/figures/mage-007/. The pair in each row was measured in one session;
the Triton and cuTile bars are references from their own sessions and are drawn
differently so the two kinds cannot be confused.

Run: uv run --script scripts/plot-kernel-lane-fixes.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mage-007"
SOURCE = ROOT / "docs/assets/results" / EXPERIMENT / "kernel-lane-fixes.json"
OUT = ROOT / "docs/assets/figures" / EXPERIMENT

BG, INK, MUTED, RULE = "#101217", "#edf0f5", "#a0a9b9", "#303641"
KIND_COLORS = {"before": "#8b93a7", "after": "#91dbba", "rejected": "#e08a8a", "reference": "#3d4759"}
KIND_EDGE = {"before": "#a0a9b9", "after": "#91dbba", "rejected": "#e08a8a", "reference": "#93caff"}


def plot(data, mobile):
    changes = data["changes"]
    width, height = (3.65, 6.6) if mobile else (7.4, 6.2)
    fig, axes = plt.subplots(len(changes), 1, figsize=(width, height))
    if len(changes) == 1:
        axes = [axes]
    fig.subplots_adjust(top=.845 if not mobile else .86, bottom=.135 if not mobile else .16,
                        left=.30 if not mobile else .40, right=.985, hspace=.72)

    fig.text(.01, .978, "What three kernel changes measured", color=INK,
             fontsize=13 if mobile else 15, weight="bold", va="top")
    fig.text(.01, .945, "GPU kernel time, before and after; references from other sessions",
             color=MUTED, fontsize=8.4 if mobile else 10, va="top")
    fig.text(.01, .012, "RTX 4090 · FP32 · each row has its own scale · pairs measured in one session",
             color=MUTED, fontsize=7.4 if mobile else 9, va="bottom")

    for ax, change in zip(axes, changes):
        bars = change["bars"]
        values = [b["us"] for b in bars]
        extent = max(values)
        ax.set_xlim(0, extent * (1.34 if mobile else 1.22))
        ax.set_ylim(len(bars) - .45, -.6)
        for y, bar in enumerate(bars):
            color = KIND_COLORS[bar["kind"]]
            ax.barh(y, bar["us"], height=.54, color=color, zorder=3,
                    edgecolor=KIND_EDGE[bar["kind"]], linewidth=1.1 if bar["kind"] == "reference" else 0)
            if bar["kind"] == "reference":
                ax.barh(y, bar["us"], height=.54, fill=False, hatch="////",
                        edgecolor=KIND_EDGE["reference"], linewidth=0, zorder=4)
            label = f"{bar['us']:.2f}" if bar["us"] < 10 else f"{bar['us']:.1f}"
            ax.annotate(label, (bar["us"], y), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=8.4 if mobile else 10)
        names = [f"{b['name']}\n(before)" if b["kind"] == "before" else b["name"] for b in bars]
        ax.set_yticks(range(len(bars)), names, color=MUTED, fontsize=7.6 if mobile else 9.4)
        for tick, bar in zip(ax.get_yticklabels(), bars):
            tick.set_color(KIND_EDGE[bar["kind"]] if bar["kind"] in ("after", "rejected") else MUTED)
        ax.set_title(change["label"], loc="left", color=INK, fontsize=9.8 if mobile else 11.5,
                     pad=14, weight="medium")
        caption = change["note"]
        if change.get("kind_note"):
            caption = f"{change['kind_note']} — {caption}"
        ax.text(0, 1.02, caption, transform=ax.transAxes, color=MUTED,
                fontsize=7 if mobile else 8.4, va="bottom")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if mobile else 5))
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.4 if mobile else 8.6)
        ax.grid(axis="x", color=RULE, linewidth=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(BG)

    fig.patch.set_facecolor(BG)
    name = f"kernel-lane-fixes{'-mobile' if mobile else ''}"
    metadata = {"Date": None, "Description":
                f"RTX 4090; {data['instrument']}. {data['session']} "
                "Hatched bars are references from other sessions. "
                "See docs/assets/results/mage-007/kernel-lane-fixes.json for the values."}
    fig.savefig(OUT / f"{name}.svg", metadata=metadata)
    svg = OUT / f"{name}.svg"
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")
    if not mobile:
        fig.savefig(OUT / f"{name}.png", dpi=200, metadata=metadata)
    plt.close(fig)


def main():
    data = json.loads(SOURCE.read_text())
    for change in data["changes"]:
        change["kind_note"] = ""
        if any(b["kind"] == "rejected" for b in change["bars"]):
            change["kind_note"] = "an attempt that measured worse and was reverted"
        elif all(b["us"] for b in change["bars"]) and not math.isnan(change["bars"][0]["us"]) \
                and change["bars"][1]["us"] < change["bars"][0]["us"]:
            ratio = change["bars"][0]["us"] / change["bars"][1]["us"]
            change["kind_note"] = f"{ratio:.2f}x faster"
    OUT.mkdir(parents=True, exist_ok=True)
    plot(data, mobile=False)
    plot(data, mobile=True)
    print("wrote", OUT / "kernel-lane-fixes.svg")


if __name__ == "__main__":
    main()
