---
title: Wider loads helped two kernels and hurt a third
permalink: /experiments/mage-007/
eyebrow: "Field note 007 / Mathematics on a GPU"
description: The same treatment — more data per thread, loaded in one instruction — closed one gap and halved another, and then made a third kernel 2.6x slower. All three were measured the same way, and the replacement was reverted.
math: true
---
The [previous note]({{ '/experiments/mage-006/' | relative_url }}) ended with the Rust kernels
at 68.6 µs of GPU kernel time for the matrix multiply and 9.0 µs for layer normalization, and
with three places where they were still behind. The worst was layer normalization **at wide
rows**: at 4096×4096 it was 1.41× slower than Triton's kernel, because the shape fell back to
an older kernel that handles one row at a time. Two more of the five operations, bias + GELU
and neighbor aggregation, had never been touched since the first comparison.

All three want the same kind of help. A thread that handles four values at once loads them
with one instruction instead of four, and measures have shown that shape winning before: it is
what took layer normalization from 18.6 µs to 11.0 µs, and what made the matrix multiply's
shared reads cheap. So the natural next move was to widen what each thread handles in all
three kernels and measure what happened.

Two of the three got faster, and the third got much slower. **Neighbor aggregation** went from
9.45 µs to 7.06 µs once each thread took a 128-bit quad of features and two edge gathers
overlapped. **Layer normalization at wide rows** turned out not to need a rewrite at all: the
two-warp kernel that serves narrower rows was already sound at 4096 and was simply excluded by
a stale size limit, so removing the limit took it from 255.41 µs to 186.80 µs. And **bias +
GELU**, given exactly the same treatment, measured 27.00 µs against the 10.29 µs it already
had — 2.6× slower — and the change was reverted.

That third result is the reason this note exists. Widening a thread's load is not a rule that
transfers between kernels; it trades instruction count for occupancy, and it pays only when the
kernel is short of instructions. The elementwise kernel already has enough threads to saturate
the device, so giving each of them four times the work only removed parallelism.

The gaps that remain are smaller: neighbor aggregation is 1.2× behind Triton, and bias + GELU
is unchanged at 11.0 µs against Triton's 7.8.

## Method

| | |
| --- | --- |
| Instrument | GPU kernel time, one Nsight Systems capture per measurement, mean of 100 launches after 25 warm-up launches |
| Host | RTX 4090 (sm_89), driver 591.74, WSL2 Ubuntu 22.04, device idle |
| Session discipline | Each before/after pair was measured in one session, with the same inputs and the same warm-up, on the same binary path; the device was checked idle before each pass |
| Correctness | Full-output check against PyTorch FP32 with TF32 disabled, `rtol = atol = 1e-4`, before any timing was believed |
| References | Triton and cuTile bars are quoted from their own sessions (mage-004, mage-006) and are drawn hatched in the figure because they are not pairings |

The figure is [`docs/assets/figures/mage-007/kernel-lane-fixes.svg`](../assets/figures/mage-007/kernel-lane-fixes.svg),
drawn by `scripts/plot-kernel-lane-fixes.py` from
[`docs/assets/results/mage-007/kernel-lane-fixes.json`](../assets/results/mage-007/kernel-lane-fixes.json).

## Values

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-007/kernel-lane-fixes-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-007/kernel-lane-fixes.svg' | relative_url }}" width="740" height="620"
         alt="Three rows of horizontal bars, GPU kernel time in microseconds, each row before and after one change. LayerNorm at 4096 by 4096: the single-warp kernel 255.41 becomes the two-warp kernel 186.80, with Triton at 181.59 for reference. Neighbor aggregation at 4096 by 64 by 65536: the scalar kernel 9.45 becomes the feature-quad kernel 7.06, with Triton at 5.97 for reference. Bias plus GELU at 4096 by 768: the scalar kernel stays at 10.29 and the feature-quad variant that measured 27.00, 2.6 times slower, was reverted, with Triton at 7.76 and cuTile Rust at 7.97 for reference.">
  </picture>
  <figcaption>
    <p>GPU kernel time before and after each change, one Nsight Systems capture of 100 launches per bar, mean. Each before-and-after pair was measured in one session on an idle device; the Triton and cuTile bars are drawn hatched because they are quoted from their own sessions (mage-006 and mage-004) and are references, not pairings. Lower is better.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-007/kernel-lane-fixes.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-007/kernel-lane-fixes.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-kernel-lane-fixes.py">Script ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs of GPU kernel time)</summary>
      <table>
        <caption class="visually-hidden">GPU kernel time before and after each kernel-lane change</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Reference</th></tr></thead>
        <tbody>
          <tr><th scope="row">LayerNorm 4096×4096</th><td>255.41</td><td>186.80</td><td>Triton 181.59</td></tr>
          <tr><th scope="row">Neighbor aggregation 4096×64×65536</th><td>9.45</td><td>7.06</td><td>Triton 5.97</td></tr>
          <tr><th scope="row">Bias + GELU 4096×768</th><td>10.29 (kept)</td><td>27.00 (reverted)</td><td>Triton 7.76, cuTile 7.97</td></tr>
        </tbody>
      </table>
    </details>
  </figcaption>
</figure>
</div>

Kernel time in microseconds, mean of 100 launches.

| Operation | Before | After | Change | Reference (other session) |
| --- | ---: | ---: | ---: | --- |
| LayerNorm 4096×4096 | 255.41 (`layer_norm_warp`) | **186.80** (`layer_norm_pair`) | 1.37× faster | Triton 181.59 (mage-006) |
| Neighbor aggregation 4096×64×65536 | 9.45 (scalar) | **7.06** (feature quad) | 1.34× faster | Triton 5.97, PyTorch 120.93 (mage-006) |
| Bias + GELU 4096×768 | 10.29 (scalar, kept) | 27.00 (feature quad, **rejected**) | 2.6× slower | Triton 7.76, cuTile 7.97 (mage-004) |

Worst full-output error: 4.77e-07 (LayerNorm), 3.58e-07 (neighbor), 5.31e-06 (GELU).

## What each change was

**LayerNorm at wide rows** (issue #54, PR #69). The two-warp row kernel was gated at
`width <= 2048`, so a 4096-wide row fell back to `layer_norm_warp` — the capture named
the kernel, which is how the fallback was confirmed rather than assumed. The pairwise
kernel's own soundness condition, each warp's span a multiple of 32 lanes, already
held at 4096: the cap was a stale heuristic. Raising it to 4096 is the whole change.

**Neighbor aggregation** (issue #59, PR #71). The scalar kernel gives each thread one
feature and a serial edge loop, so one gather is in flight per thread. The quad form
gives each thread four features — one 128-bit load per edge — and unrolls the edge
walk by two, so two independent gathers overlap. Triton's 5.97 µs remains ahead, so
the gap narrows from 1.7× to 1.2×; the kernel moves about 17 MB of gathered rows in
7 µs, which is L2-bandwidth work on this part, so more edges in flight per thread is
the next lever.

**Bias + GELU** (issue #59, PR #71). The same vectorization was tried and rejected:
four features per thread measured 27.00 µs against the scalar kernel's 10.29 µs in the
same session. The LayerNorm treatment does not transfer to this elementwise shape —
one element per thread wins, most likely on occupancy. Kept here because it is the
counterexample: "vectorize the elementwise kernel" is not a rule.

## What the numbers do not establish

- No hardware counters are available, so the L2-bandwidth reading of the neighbor
  kernel is inferred from traffic arithmetic, not measured.
- One capture per point: a kernel time here carries no interval of its own.
- The references are quoted from other sessions. A consolidated pass
  (`scripts/evolve_capture.py --all`) runs every implementation in one session, and
  running it would make these four arms a single measurement.
- The LayerNorm change is validated at 4096×4096, 4096×512, 3072×1024 and 2048×2048;
  widths above 4096 still fall back, and were not measured.
- GELU's rejection is a single before/after pair, not a sweep: other vector widths
  (two features, or quads with a different block size) were not tried.

## Reproduction

```bash
source scripts/oxide-env.sh
cd examples/oxide && CARGO_BUILD_JOBS=4 cargo oxide build --arch sm_89 && cd ../..

# inputs in the harness layout, at the shape under test
.venv/bin/python -c "
import sys; sys.path.insert(0, 'examples/oxide')
import experiment
from pathlib import Path
experiment.generate(Path('artifacts/mage-007/layernorm-4096x4096'), 'layernorm', [4096, 4096],
                    warmup=25, iterations=100)
"

# one operation, one capture; the kernel time is in the report's kernels.json
.venv/bin/python -m mage profile-exec --backend nsys --capture-range cuda   --output-dir artifacts/mage-007/layernorm-4096x4096/nsys --   examples/oxide/target/release/mage-oxide artifacts/mage-007/layernorm-4096x4096   --iterations 100 --capture

uv run --script scripts/plot-kernel-lane-fixes.py
```
