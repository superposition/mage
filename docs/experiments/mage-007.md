---
title: Wider loads helped two kernels and hurt a third
permalink: /experiments/mage-007/
eyebrow: "Field note 007 / Mathematics on a GPU"
description: "A wider load per thread is a trade: fewer instructions issued, more bytes in flight. Applied to three kernels it paid twice and cost once, and the one it cost is the one that was already using every thread the device has."
math: true
mesh_band: true
---
{% include mesh-band.html
   colors="#91dbba,#93caff,#e08a8a"
   weights="1.34,1.37,0.38"
   still="/assets/figures/mage-007/mesh-band.png"
   alt="A dark field with three soft spots of colour — green, blue and muted red — the red one much dimmer than the other two."
   caption="The three spots are this note's three changes, opacity set by the speed-up each measured." %}**The claim.** Making each thread load four values at once instead of one is a memory decision, not
an optimisation: it cuts the number of load instructions by four and puts $4\times$ the bytes in
flight per instruction. It pays when a kernel is short of instructions to issue. It costs when the
grid already fills the machine and the only thing hiding memory latency is how many threads are
running. The same change, applied to three kernels, gave $1.34\times$ and $1.37\times$ speedups,
and then a $2.6\times$ slowdown.

## Argument one: a gather wants fewer, wider loads

Neighbor aggregation computes one output row per input row:

$$y_{i,f} \;=\; \sum_{e \in \mathrm{row}(i)} w_e \, x_{i_e,\, f}$$

The edge list is data — row $i$ owns `rowptr[i]..rowptr[i+1]` entries of `indices` — so each thread
walks its own row's edges and asks global memory for one value of $x$ per edge. At
$4096 \times 64$ with $65536$ edges each thread issues 16 loads of 32 bits, one at a time, and each
one's address depends on the previous edge's index. There is almost nothing to overlap.

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-007/neighbor-access-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-007/neighbor-access.svg' | relative_url }}" width="740" height="270"
         alt="Two schematics of one thread's memory traffic. Before, one feature per thread: a 32-bit load per edge, one edge in flight. After, four features per thread in a single 128-bit load, two edges unrolled and in flight.">
  </picture>
  <figcaption>
    <p>What one thread reads before and after. Drawing, not a measurement — the load widths and the edges in flight come from the two kernels' code. Four features per thread makes one 128-bit load; unrolling the edge walk by two keeps two of those loads in flight at once.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-007/neighbor-access.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-007/neighbor-access.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-neighbor-access.py">Script ↗</a>
    </div>
  </figcaption>
</figure>
</div>

Four features per thread turns each of those loads into a 128-bit quad, and unrolling the walk by
two gives the memory system two independent requests instead of a dependency chain. Measured across
100 launches: $9.45 \to 7.06\ \mu s$. Triton's kernel, which loads $32$ edges at once into a tile
and reduces over them, is still ahead at $5.97\ \mu s$: the gathered rows are about
$65536 \times 256\ \mathrm{B} = 16.8\ \mathrm{MB}$, which is L2 traffic on this part, so the only
thing left to win is more requests in flight.

## Argument two: sometimes nothing needs to change

Layer normalization computes, per row:

$$y_f \;=\; \gamma_f \, \frac{x_f - \mu}{\sqrt{\sigma^2 + \epsilon}} \;+\; \beta_f$$

and the kernel for it holds the row in registers, so the split of the row across warps is fixed by
size. At 4096-wide rows the kernel that handles them splits each row across two warps and the two
partial sums meet in 64 bytes of shared memory behind one barrier; it was gated at width 2048, so
$4096$ fell through to a kernel that walks the whole row in a single warp. The gate was stale. The
only structural requirement is that each warp's span be a whole number of 32-lane steps, and at
$4096$ each half is $2048$ elements, which is. Removing the gate measured $255.41 \to
186.80\ \mu s$, against Triton's $181.59\ \mu s$ — the largest single win of the three, from
deleting one condition.

## Argument three: the counterexample

Bias + GELU has no reuse and no gathering. It applies

$$\mathrm{gelu}(z) \;=\; \tfrac{1}{2} z \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(z + 0.044715\,z^3\right)\right)\right)$$

to $z = x + \text{bias}$, elementwise. At $4096 \times 768$ that is 3.1M elements; with one element
per thread it is 12,288 blocks of 256 threads, which already fills the device, and every thread's
work is independent. Four elements per thread cuts that to 786,432 threads and gives each thread
four `tanh` evaluations in sequence. Measured $27.00\ \mu s$ against $10.29\ \mu s$ — the change was
reverted, and the scalar kernel stayed.

That is the whole lesson of this note: **the same edit is a $1.4\times$ win on two kernels and a
$2.6\times$ loss on a third**, and what separates them is whether the kernel was short of
instructions or short of threads.

## Evidence

| Operation | before | after | reference (other session) |
| --- | ---: | ---: | --- |
| Neighbor aggregation $4096\times64\times65536$ | 9.45 | **7.06** | Triton 5.97, PyTorch 120.93 |
| LayerNorm $4096\times4096$ | 255.41 | **186.80** | Triton 181.59 |
| Bias + GELU $4096\times768$ | 10.29 (kept) | 27.00 (reverted) | Triton 7.76, cuTile 7.97 |

*Microseconds of GPU kernel time, mean of 100 launches after 25 warm-up launches. Each before and
after pair was measured in one session on an idle device; the references come from mage-004 and
mage-006 and are hatched in the figure below because they are not pairings. Worst full-output error
against PyTorch: $4.8\times10^{-7}$ (LayerNorm), $3.6\times10^{-7}$ (neighbor), $5.3\times10^{-6}$
(GELU).*

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-007/kernel-lane-fixes-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-007/kernel-lane-fixes.svg' | relative_url }}" width="740" height="620"
         alt="Three rows of horizontal bars of GPU kernel time in microseconds, each row before and after one change. Neighbor aggregation falls from 9.45 to 7.06 with Triton at 5.97 for reference. LayerNorm at 4096 by 4096 falls from 255.41 to 186.80 with Triton at 181.59. Bias plus GELU keeps its scalar kernel at 10.29 while the feature-quad variant that measured 27.00 was reverted, with Triton at 7.76 and cuTile Rust at 7.97.">
  </picture>
  <figcaption>
    <p>The three changes as bars, with the other-session references hatched. Lower is better.</p>
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

## What the numbers do not establish

- No hardware counters are available, so "$16.8$ MB of L2 traffic" is arithmetic on the bytes moved,
  not a measurement.
- One capture per point: a kernel time carries no interval of its own.
- The references come from other sessions; `scripts/evolve_capture.py --all` runs every
  implementation in one pass and has not been run on these shapes.
- The LayerNorm gate is only validated where the harness points it: $4096\times4096$,
  $4096\times512$, $3072\times1024$, $2048\times2048$. Widths above 4096 still fall through.
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
