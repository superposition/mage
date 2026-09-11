---
title: What the compiler chose
permalink: /experiments/mage-004/
eyebrow: "Field note 004 / Mathematics on a GPU"
description: Handing the kernel decisions to a tile compiler: where it beat the hand-written kernels, where it lost, and why most of the apparent difference turned out to be the cost of starting work rather than the cost of doing it.
math: true
---

The [earlier notes]({{ '/experiments/mage-003/' | relative_url }}) wrote the kernels by hand. Each
thread owned a small block of results, and the layout of shared memory, the width of each load and
the placement of barriers were all deliberate choices that produced measured wins.

This note hands those choices to a compiler. [cuTile Rust](https://github.com/NVlabs/cutile-rs)
lets you write a kernel as a single-threaded program over *tiles* — blocks of data — and works out
the threads, the memory layout and the tensor-core instructions itself. The question is what that is
worth on the same five operations, measured the same way.

The answer has three parts, and the middle one is the surprise.

**The compiler is competitive.** It beats the hand-written kernel on bias + GELU (8.19 µs against
11.0) and draws level on layer normalization (10.8 against 10.1). It loses on the two
matrix-shaped operations, 1.6× and 1.4×, and by 3.4× on neighbor aggregation, which is irregular
enough that the tile model has no safe way to express it.

**Most of the gap in the timing column was not the kernel.** Timing that waits for every call to
finish prices the *cost of starting work*, and this runtime starts work expensively: 14–23 µs per
call against 2–3 µs for the hand-written kernels. Queue the calls up, or replay them from a
recorded graph, and the numbers fall onto the kernel times. The kernels were never as far apart as
the column said, and any comparison that blocks on every call will mislead in the same direction.

**The library's own search beat mine.** Twelve hand-picked tile shapes moved the matrix multiply
from 738 µs to 201 µs of measured time; the bundled autotuner then found a shape that measured
137 µs against my 189 µs. Picking twelve configurations to trace a ratio is not a search.

## The five operations, two views

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust |
| --- | ---: | ---: | ---: | ---: |
| Matrix multiplication 1024³ | 56.04 | 83.06 | 131.56 | **80.00** |
| Bias + GELU 4096×768 | 15.88 | 7.73 | **8.19** | 11.0 |
| LayerNorm 4096×768 | 11.42 | 8.15 | 10.76 | 10.05 |
| Triangle contraction 128×32 | 28.64 | 102.19 | 116.08 | 80.1 |
| Neighbor aggregation 4096×64×65536 | 67.32 | 7.97 | 35.37 | 10.3 |

*Microseconds of GPU kernel time per operation, from one Nsight Systems capture of 100 launches
each. Lower is better. PyTorch's bias + GELU is two kernels, its triangle contraction three and
its neighbor aggregation four; every other cell is one.*

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-004/comparison-kernel-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-004/comparison-kernel.svg' | relative_url }}" width="740" height="468"
         alt="Five horizontal bar charts, one per operation, of GPU kernel time in microseconds for PyTorch, Triton, the cuTile Rust tile kernels and the cuda-oxide Rust kernels. Bias plus GELU is shortest for Triton at 7.73, then cuTile at 8.19, cuda-oxide at 11.0 and PyTorch at 15.88. Matrix multiplication is shortest for PyTorch at 56.04 and longest for cuTile at 131.56. Neighbor aggregation is shortest for Triton at 7.97 and longest for PyTorch at 67.32.">
  </picture>
  <figcaption>
    <p>GPU kernel time per operation, from separate Nsight Systems captures of 100 launches each. Each row has its own scale. The tile kernels win bias + GELU, draw on layer normalization, and trail on the other three.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-004/comparison-kernel.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-004/comparison-kernel.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-comparison.py">Script ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs of GPU kernel time)</summary>
      <table>
        <caption class="visually-hidden">GPU kernel time per operation and implementation</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">PyTorch</th><th scope="col">Triton</th><th scope="col">cuTile Rust</th><th scope="col">cuda-oxide Rust</th></tr></thead>
        <tbody>
          <tr><th scope="row">Matrix multiplication 1024³</th><td>56.04</td><td>83.06</td><td>131.56</td><td>80.00</td></tr>
          <tr><th scope="row">Bias + GELU 4096×768</th><td>15.88</td><td>7.73</td><td>8.19</td><td>11.0</td></tr>
          <tr><th scope="row">LayerNorm 4096×768</th><td>11.42</td><td>8.15</td><td>10.76</td><td>10.05</td></tr>
          <tr><th scope="row">Triangle contraction 128×32</th><td>28.64</td><td>102.19</td><td>116.08</td><td>80.1</td></tr>
          <tr><th scope="row">Neighbor aggregation 4096×64×65536</th><td>67.32</td><td>7.97</td><td>35.37</td><td>10.3</td></tr>
        </tbody>
      </table>
    </details>
  </figcaption>
</figure>
</div>

The same run, timed the way most comparisons are timed — one measurement around each call, waiting
for it to finish:

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust |
| --- | ---: | ---: | ---: | ---: |
| Matrix multiplication 1024³ | 52.40 | 93.50 | 170.79 | 82.6 |
| Bias + GELU 4096×768 | 33.02 | 27.93 | 29.97 | 13.3 |
| LayerNorm 4096×768 | 21.68 | 23.52 | 37.22 | 12.9 |
| Triangle contraction 128×32 | 61.81 | 99.96 | 157.84 | 83.5 |
| Neighbor aggregation 4096×64×65536 | 95.07 | 27.00 | 61.22 | 12.9 |

*Microseconds around each call, mean of three rounds of 100 samples with the implementations
rotated. Every implementation is measured the same way here, and the ordering is different.*

## Where the difference goes

The two tables disagree, and the disagreement is measurable rather than mysterious. Taking the
same launches and queueing ten of them behind one measurement:

| Operation | awaited | queued by ten | kernel |
| --- | ---: | ---: | ---: |
| Matrix multiplication 1024³ | 150.53 | 127.07 | 131.56 |
| Bias + GELU 4096×768 | 25.76 | 8.50 | 8.19 |
| LayerNorm 4096×768 | 30.62 | 10.64 | 10.76 |
| Triangle contraction 128×32 | 136.19 | 121.80 | 116.08 |
| Neighbor aggregation 4096×64×65536 | 49.28 | 33.28 | 35.37 |

Queued, every row lands on its kernel time — bias + GELU and layer norm within 0.5 µs, and matrix
multiply below its own kernel time because submission overlaps execution. Recording ten launches
into a replayable graph does the same: bias + GELU comes out at 7.77 µs against an 8.19 µs kernel.
Timed the same way, Triton improves by 6–7 µs per call and PyTorch by about 1 µs, so with all
three launch paths matched the wins and losses are the ones in the first table and nothing else.

*Caveat: the cuda-oxide column above is the third note's retained measurement, not taken in the
same session. The kernel track has since published [mage-006]({{ '/experiments/mage-006/' | relative_url }})
with lower numbers of its own, so the two Rust columns should not be subtracted from each other.*

## The tile shape, and the search that beat me

The matrix multiply started from the upstream tutorial's 16 × 16 × 8 tile and was slow. Twelve
hand-picked configurations took it from 738 µs to 201 µs, and the shape was not monotone in any
direction: deepening the contraction step cost 42% at a 16 × 16 tile but only 6% at 64 × 64, and
128 × 128 × 8 was 2.3× *slower* than 128 × 64 × 8 — the signature of a register or occupancy
limit rather than arithmetic.

| Tile | Span | Tile | Span |
| --- | ---: | --- | ---: |
| 16×16×8 | 738.0 | 64×128×8 | 213.9 |
| 32×32×32 | 488.8 | 128×64×8 | 201.5 |
| 64×64×32 | 255.4 | 128×128×8 | 469.8 |

Then the library's autotuner searched the same space properly — 36 candidates, each validated
before timing, kernel time alone, 17 seconds — and chose **32 × 128 × 32**, which measured
137.28 µs against my 189.44 µs when both were checked in one session. Every number in this note
was re-measured with it afterwards, which is where the matrix multiply's 131.56 µs comes from.

## What the compiler would not let us write

These cost time to find, and each is silent until the compiler or the assembler runs:

- **Tile dimensions must be powers of two.** A row of 768 columns is rejected with
  `failed to compile Tile IR program` and no further detail, so layer norm pads each row to 1024
  and divides by the true width — 33% of its lanes are wasted, and it still draws level.
- **A partition load indexed by a loop variable does not vary.** The triangle kernel loaded every
  channel through a `for` loop and read the first channel every time; the channel has to come from
  the grid axes instead.
- **A `Tile<..>` written inside an expression is not rewritten** by the entry macro, so it must
  appear only in a `let` annotation.
- **`convert_scalar` has no `u32` → `i32`.** The CSR arrays are uploaded as `i32`, which the
  host's bounds check makes safe.
- **Scalar comparison is not a supported operator**, so the edge walk counts its edges and loops
  over the count.

The irregular operation needed raw device pointers (`load_ptr_tko`) to read its row pointers and
edge indices. That escape hatch is the honest boundary of the safe tile model, and it is why that
kernel is the one that trails furthest.

## What the numbers do not establish

- No hardware counters are available here, so occupancy and bandwidth are inferred from ratios,
  not read.
- Layer norm's 10.76 µs includes the 768 → 1024 row padding; a native 768-wide tile would do less
  work.
- One capture per operation: a kernel time here carries no interval of its own. The other column
  comes from three rotating rounds.
- Measured timings are sensitive to a busy device. A repeat that shared the GPU with another
  capture measured three to twenty times larger spans for the same binary, and it was caught only
  because PyTorch's own kernel moved with it.
- Compilation, transfers and process startup are excluded throughout.

## Open items

- **Replay for the other three kernels.** Only matrix multiply and bias + GELU can be replayed
  from a recorded graph today; each kernel needs its own capture.
- **Neighbor aggregation stays 3.4× behind.** It gathers about 17 MB of rows, which is
  L2-bandwidth work on this part, and its 35.37 µs is far above what that traffic accounts for.
- **Lower precision is a separate question.** FP16, BF16 and TF32 have their own error budgets and
  nothing here speaks to them.

## Reproduction

```bash
source scripts/cutile-env.sh
cd examples/cutile && cargo build --release && cd ../..
.venv/bin/python examples/oxide/comparison.py --implementation cutile \
  --experiment mage-004 --output artifacts/mage-004-comparison \
  --rounds 3 --iterations 100 --warmup 25
```

`CUTILE_MATMUL_TILE=BM,BN,BK` picks a different tile; `--mode single|batch|graph` picks the launch
path; `--features tune` adds the autotuner. Run one device experiment at a time.
