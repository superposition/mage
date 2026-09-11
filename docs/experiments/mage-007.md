---
title: Wider loads helped two kernels and hurt a third
permalink: /experiments/mage-007/
eyebrow: "Field note 007 / Mathematics on a GPU"
description: "One gap closed, one halved, and one attempt that measured 2.6x slower and was reverted, each before-and-after pair taken in one session."
math: true
---
Neighbor aggregation sums, for each output row, the input rows named by that row's edges. Each
thread read one float per edge — a 32-bit load, one edge at a time — so a row with 16 edges issued
16 narrow loads in a dependent chain:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-007/neighbor-access-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-007/neighbor-access.svg' | relative_url }}" width="740" height="270"
         alt="Two schematics of one thread's memory traffic. Before: one feature per thread, a 32-bit load per edge, one edge in flight. After: four features per thread in a single 128-bit load, two edges unrolled and in flight.">
  </picture>
  <figcaption>
    <p>What one thread reads in the neighbor kernel, before and after. Drawing, not a measurement: the load widths and the edges in flight are read from the two kernels' code.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-007/neighbor-access.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-007/neighbor-access.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-neighbor-access.py">Script ↗</a>
    </div>
  </figcaption>
</figure>
</div>

Four features per thread makes that one 128-bit load, and unrolling the edge walk by two puts two
gathers in flight. Measured 9.45 → 7.06 µs across 100 launches at 4096×64×65536. Triton's kernel,
which loads 32 edges at once into a tile, is still ahead at 5.97 µs; the gathered rows are about
17 MB, which is L2 traffic on this part, so more edges in flight is what is left.

Layer normalization at 4096-wide rows needed no rewrite. The kernel for these rows splits each row
across two warps, and the two partial sums meet in 64 bytes of shared memory behind one barrier.
It was gated at width 2048, so 4096 fell through to a kernel that walks the row in a single warp.
The gate was stale: the only structural requirement is that each warp's span be a whole number of
32-lane steps, and at 4096 each half is 2048 elements, which is. Removing the gate measured
255.41 → 186.80 µs, against Triton's 181.59.

Bias + GELU took the same treatment and lost 2.6×. The shape is 4096×768: 3.1M elements, or 12,288
blocks of 256 threads with one element each, which already fills the device. Four elements per
thread cuts that to 786,432 threads and gives each thread four serial `tanh` evaluations. Measured
27.00 µs against 10.29 µs, and reverted. Vector width pays when a thread is short of work to
issue; it costs when the grid is already saturating the machine.

| Operation | Before | After | Reference (other session) |
| --- | ---: | ---: | --- |
| Neighbor aggregation 4096×64×65536 | 9.45 | **7.06** | Triton 5.97, PyTorch 120.93 |
| LayerNorm 4096×4096 | 255.41 | **186.80** | Triton 181.59 |
| Bias + GELU 4096×768 | 10.29 (kept) | 27.00 (reverted) | Triton 7.76, cuTile 7.97 |

*Microseconds of GPU kernel time, mean of 100 launches after 25 warm-up launches. Each before and
after pair was measured in one session on an idle device; the references are quoted from mage-004
and mage-006 and are drawn hatched in the figure below because they are not pairings.*

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-007/kernel-lane-fixes-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-007/kernel-lane-fixes.svg' | relative_url }}" width="740" height="620"
         alt="Three rows of horizontal bars of GPU kernel time in microseconds, each row before and after one change. Neighbor aggregation falls from 9.45 to 7.06 with Triton at 5.97 for reference. LayerNorm at 4096 by 4096 falls from 255.41 to 186.80 with Triton at 181.59. Bias plus GELU keeps its scalar kernel at 10.29 while the feature-quad variant that measured 27.00 was reverted, with Triton at 7.76 and cuTile Rust at 7.97.">
  </picture>
  <figcaption>
    <p>The same three changes as bars, with the other-session references hatched. Lower is better.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-007/kernel-lane-fixes.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-007/kernel-lane-fixes.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-kernel-lane-fixes.py">Script ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs of GPU kernel time)</summary>
      <table>
        <caption class="visually-hidden">GPU kernel time before and after each change</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Reference</th></tr></thead>
        <tbody>
          <tr><th scope="row">Neighbor aggregation 4096×64×65536</th><td>9.45</td><td>7.06</td><td>Triton 5.97</td></tr>
          <tr><th scope="row">LayerNorm 4096×4096</th><td>255.41</td><td>186.80</td><td>Triton 181.59</td></tr>
          <tr><th scope="row">Bias + GELU 4096×768</th><td>10.29 (kept)</td><td>27.00 (reverted)</td><td>Triton 7.76, cuTile 7.97</td></tr>
        </tbody>
      </table>
    </details>
  </figcaption>
</figure>
</div>

## Limits of these numbers

- No hardware counters are available, so "L2 traffic" is arithmetic on the bytes moved, not a
  measurement.
- One capture per point: a kernel time carries no interval of its own.
- The references come from other sessions (`scripts/evolve_capture.py --all` runs every
  implementation in one session, and has not been run on these shapes).
- The LayerNorm change is only validated where the harness points it: 4096×4096, 4096×512,
  3072×1024, 2048×2048. Widths above 4096 still fall through.
- GELU's rejection is one pair, not a sweep: two features per thread, and quads with other block
  sizes, were not tried.

## Reproduce

```bash
source scripts/oxide-env.sh
cd examples/oxide && CARGO_BUILD_JOBS=4 cargo oxide build --arch sm_89 && cd ../..
.venv/bin/python -c "
import sys; sys.path.insert(0, 'examples/oxide')
import experiment
from pathlib import Path
experiment.generate(Path('artifacts/mage-007/layernorm-4096x4096'), 'layernorm', [4096, 4096],
                    warmup=25, iterations=100)
"
.venv/bin/python -m mage profile-exec --backend nsys --capture-range cuda \
  --output-dir artifacts/mage-007/layernorm-4096x4096/nsys -- \
  examples/oxide/target/release/mage-oxide artifacts/mage-007/layernorm-4096x4096 \
  --iterations 100 --capture
uv run --script scripts/plot-kernel-lane-fixes.py
uv run --script scripts/plot-neighbor-access.py
```
