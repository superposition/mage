---
title: What the compiler chose
permalink: /experiments/mage-004/
eyebrow: "Field note 004 / Mathematics on a GPU"
description: A tile compiler against hand-written threads on five FP32 operations, where the difference turned out to be the launch path rather than the kernel, and where the library's own search beat the hand-picked one.
math: true
---

The earlier notes wrote the kernels by hand: each thread owned a register tile, and
the shared-memory layout and the barriers were chosen deliberately. This note hands
those decisions to a compiler. [cuTile Rust](https://github.com/NVlabs/cutile-rs)
takes *tile* programs — single-threaded code over tiles — and maps them onto warps,
blocks, shared memory and tensor cores through CUDA Tile IR. The three measured wins
of the [third field note]({{ '/experiments/mage-003/' | relative_url }}) were tiling
decisions, so the question here is what happens when the compiler makes them.

It makes them well enough to beat the hand-written kernel on one operation and to
match it on another, and it does not on the two matmul-shaped ones. But the more
useful result is elsewhere: **most of the difference in the comparison's timing
column was not the kernel at all.** A lazy runtime prices its host submission path
when every call is awaited, and the tile runtime's costs about 16 µs per launch
against the hand-written runtime's 2–3 µs. Batch the launches or replay them from a
CUDA graph and the spans fall onto the kernel times. The kernels were never as far
apart as the column said.

The journal entry for this work, *Tiles the compiler chose*, is published at the
[Superposition journal](https://superposition.github.io/journal/). The earlier
rounds are [field note 001]({{ '/experiments/mage-001/' | relative_url }}),
[002]({{ '/experiments/mage-002/' | relative_url }}) and
[003]({{ '/experiments/mage-003/' | relative_url }}).

## The five operations, two views

Five forward FP32 operations run through the same harness as the earlier rounds:
identical inputs and hashes, every output checked against PyTorch with TF32
disabled, and two independent views of the cost — GPU kernel time from separate
Nsight Systems captures, and the span around the call from CUDA events.

## Method

| | |
| --- | --- |
| Implementation | `examples/cutile`, crate `mage-cutile`, `cutile = "=0.3.1"` from crates.io |
| Toolchain | Rust 1.98.1, CUDA 13.3 toolkit (`/usr/local/cuda-13.3`) |
| Host | RTX 4090 (sm_89), driver 591.74, WSL2 Ubuntu 22.04 |
| Inputs | The same generated files, seeds and SHA-256 hashes as mage-001..003 |
| Correctness | Every output element against PyTorch FP32 with TF32 disabled, `rtol = atol = 1e-4` |
| Event spans | Per-launch CUDA event pairs, 25 warmup launches, 100 samples per round, three rounds with rotating implementation order |
| Kernel time | One Nsight Systems capture per operation, 100 launches, `--capture-range cuda` |

One specialization is compiled at the first launch of each kernel and reused for
every measured launch, so compilation is outside the timed region.

## Event spans (µs, mean of three rotating rounds)

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust (mage-003) |
| --- | --- | --- | --- | --- |
| Matrix multiplication 1024³ | 52.40 | 93.50 | 170.79 | 82.6 |
| Bias + GELU 4096×768 | 33.02 | 27.93 | 29.97 | 13.3 |
| LayerNorm 4096×768 | 21.68 | 23.52 | 37.22 | 12.9 |
| Triangle contraction 128×32 | 61.81 | 99.96 | 157.84 | 83.5 |
| Neighbor aggregation 4096×64×65536 | 95.07 | 27.00 | 61.22 | 12.9 |

## GPU kernel time (µs per operation, single capture, 100 calls)

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust (mage-003) |
| --- | --- | --- | --- | --- |
| Matrix multiplication 1024³ | 56.04 | 83.06 | 131.56 | 80.00 |
| Bias + GELU 4096×768 | 15.88 | 7.73 | **8.19** | 11.0 |
| LayerNorm 4096×768 | 11.42 | 8.15 | 10.76 | 10.05 |
| Triangle contraction 128×32 | 28.64 | 102.19 | 116.08 | 80.1 |
| Neighbor aggregation 4096×64×65536 | 67.32 | 7.97 | 35.37 | 10.3 |

Kernel time is summed over the launches one operation needs — PyTorch's bias +
GELU is two kernels, its triangle contraction three and its neighbor
aggregation four; every other cell is one. The launch counts come from
`comparison-profiles.json`, not from a count of what the operation *should*
launch.

The two views disagree, and the disagreement is the result. On GPU time the tile
kernels are **ahead of cuda-oxide on bias + GELU** (8.19 against 11.0 µs) and
level with it on layer norm (10.76 against 10.05 µs), while trailing on the
other three: 1.6× on matrix multiply, 1.4× on triangle contraction and 3.4× on
neighbor aggregation. On event spans the tile kernels trail everywhere,
including where their kernels are faster: bias + GELU spans 29.97 µs around a
kernel that takes 8.19 µs.

That gap is the launch path, not the kernel. Each timed iteration here records
an event, launches, records a second event and synchronizes; for a runtime whose
device operations are lazy, that pattern serializes submission and measures it.
The cuda-oxide binary launches a driver kernel directly.

### The launch path, measured

The same launches, awaited one at a time versus queued ten at a time behind one
event pair (`--mode single` and `--mode batch --batch 10`), median µs per launch,
100 samples after 25 warmup launches, beside the kernel time from the capture
above:

| Operation | single | batch:10 | kernel |
| --- | --- | --- | --- |
| Matrix multiplication 1024³ | 150.53 | 127.07 | 131.56 |
| Bias + GELU 4096×768 | 25.76 | 8.50 | 8.19 |
| LayerNorm 4096×768 | 30.62 | 10.64 | 10.76 |
| Triangle contraction 128×32 | 136.19 | 121.80 | 116.08 |
| Neighbor aggregation 4096×64×65536 | 49.28 | 33.28 | 35.37 |

The difference between the first two columns is what the host spent per launch
while every launch was awaited: 14–20 µs on the four short operations and 23 µs
on matrix multiply. Batched, the spans converge on the kernel time — bias + GELU
and layer norm land within 0.5 µs of their kernels, and matrix multiply's batched
span sits *below* its kernel time because submission overlaps execution.

The cuda-oxide spans in the event table (13.3 µs for bias + GELU over an 11.0 µs
kernel) put that runtime's per-launch host cost at 2–3 µs. So on these shapes the
tile runtime's host path is roughly 7× the SIMT one's, and the event-span column
above is inflated by exactly that difference. That is the answer to the question
the second field note left open, for the span view: the native advantage in the
span column is a launch-path artifact, not a kernel one.

The batch sweep does not improve monotonically on matrix multiply (batch 1 → 178.2,
2 → 173.6, 5 → 173.9, 10 → 160.5, 25 → 184.5, 100 → 185.0 median µs): at this
tile shape that kernel is long enough to hide submission, so batching only
matters where the kernel is short.

### Replay, and the end of the launch-path gap

The third mode captures N kernel nodes into one CUDA graph and replays it behind
a single event pair (`--mode graph --batch N`, implemented for matrix multiply
and bias + GELU — the kernels that bracket the range). Median µs per kernel
execution, same 100 samples after 25 warmup launches:

| Operation | single | batch:10 | batch:100 | graph:10 | graph:25 | kernel |
| --- | --- | --- | --- | --- | --- | --- |
| Matrix multiplication 1024³ | 138.24 | 133.02 | 140.48 | **122.87** | 144.11 | 131.56 |
| Bias + GELU 4096×768 | 25.60 | 8.40 | 10.17 | 8.73 | **7.77** | 8.19 |

Replay is the best mode for both: bias + GELU lands at 7.77 µs against an 8.19 µs
kernel, so the host cost that dominated the single-launch column is gone
entirely, and matrix multiply's replayed span sits below its kernel time because
capture removes per-launch submission from the critical path. Batching and
replay are within noise of each other on these shapes; what both establish is
that the span column's tile-runtime penalty is a submission artifact rather than
a property of the kernels.

### Matched launch paths, all three implementations

The comparison's span column awaits every call, for every implementation. Timing
PyTorch and Triton batched by ten the same way, median µs per call, 100 samples
after 25 warmup calls:

| Operation | PyTorch single → batched | Triton single → batched | cuTile single → batched |
| --- | --- | --- | --- |
| Matrix multiplication 1024³ | 50.18 → 46.39 | 87.04 → 75.82 | 139.09 → 122.26 |
| Bias + GELU 4096×768 | 18.43 → 19.35 | 19.46 → 13.52 | 25.60 → 8.50 |
| LayerNorm 4096×768 | 13.31 → 12.08 | 20.48 → 13.62 | 30.66 → 10.65 |
| Triangle contraction 128×32 | 46.08 → 35.43 | 89.09 → 78.52 | 131.07 → 112.33 |
| Neighbor aggregation 4096×64×65536 | 84.00 → 68.20 | 20.48 → 13.93 | 51.39 → 33.28 |

Batching helps every implementation, and helps the tile runtime most because its
per-call host cost was the largest. With the launch path matched, bias + GELU
(8.50) and layer norm (10.65) become the tile kernels' wins, matrix multiply and
triangle contraction stay behind because those kernels are genuinely slower
(131.56 and 116.08 against 56.04 and 28.64), and neighbor aggregation lands
between PyTorch and Triton. Triton's own launcher costs 6–7 µs per awaited call
(19.46 → 13.52 on bias + GELU); PyTorch's costs about 1 µs.

## Correctness

Every value in both tables passed the full-element check on every round. Worst
observed error:

| Operation | max abs error | tolerance |
| --- | --- | --- |
| Matrix multiplication | 1.38e-05 | 1e-4 |
| Bias + GELU | 2.38e-07 | 1e-4 |
| LayerNorm | 4.77e-07 | 1e-4 |
| Triangle contraction | 7.15e-07 | 1e-4 |
| Neighbor aggregation | 2.98e-07 | 1e-4 |

The matrix multiply is the interesting one: `mma` on `f32` is not the tensor-core
TF32 path that the same intrinsic selects in a lower-precision kernel. Its error
against the strict-FP32 reference is 1.5e-05 on K = 1024, which is FP32
accumulation error, not a 10-bit mantissa's.

## The tile shape chose the matmul result

The matmul kernel's first tile was the upstream tutorial's 16 × 16 × 8, and it
was slow. Twelve measured configurations of `BM × BN × BK`, each a separate
specialization, all producing the same output, event span on 1024³ (µs):

| Tile | Span | Tile | Span |
| --- | --- | --- | --- |
| 16×16×8 | 738.0 | 64×64×32 | 255.4 |
| 16×16×32 | 1072.4 | 64×128×8 | 213.9 |
| 32×32×8 | 526.1 | **128×64×8** | **201.5** |
| 32×32×32 | 488.8 | 128×64×16 | 246.4 |
| 64×64×8 | 271.1 | 128×64×32 | 342.9 |
| 128×128×8 | 469.8 | 256×64×8 | 612.0 |

The retained tile from this sweep is 128 × 64 × 8, 3.7× faster than the starting
one — and the autotuner below then beat it, so the binary ships 32 × 128 × 32 and
every comparison table in this note was measured with that. The sweep table stands
as the record of the hand-picked search. The shape
is not monotone in any single dimension: deepening K from 8 to 32 costs 42%
at 16 × 16 but only 6% at 64 × 64, and 128 × 128 × 8 is 2.3× slower than
128 × 64 × 8, which is the signature of a register or occupancy cliff rather
than of arithmetic. This is the same class of result the thread-level kernels
reported in the third note: the ratio between two builds is what identifies the
limit when hardware counters are unavailable.

### The autotuner disagreed, and was right

`--tune` (built with `--features tune`) runs `cutile::tune` over the same powers
of two: 36 candidates of `BM ∈ {16, 32, 64, 128} × BN ∈ {32, 64, 128} ×
BK ∈ {8, 16, 32}`, each gated by one correctness launch, measured with the
library's own device-event timing, and finished by a paired A/B runoff between
the two finalists. It chose **32 × 128 × 32**, and its trial log is retained
under `docs/assets/results/mage-004/tuning/`.

The two searches optimized different things. The hand-picked sweep above measured
the harness's *event span*, which includes the host submission path; the tuner
measured *kernel time* alone. Cross-checked afterwards in one clean session with
the harness, on the same inputs:

| Tile | single-launch span | batched by ten |
| --- | --- | --- |
| 128 × 64 × 8 (hand-picked) | 189.44 | 174.90 |
| **32 × 128 × 32 (tuner)** | **137.28** | **121.75** |

The tuner's pick is faster in both views, so the hand-picked ladder — twelve
configurations, chosen to trace a ratio rather than to search — simply missed
the better region. The library's search found it in 17 seconds and 38 trials.

**The binary ships 32 × 128 × 32**, and every table in this note was re-measured
with it in a single session on an idle device (utc 01:19). The tuned tile moved
matrix multiply from 172.52 to **131.56 µs** of kernel time and from 218.50 to
170.79 µs of span, which is where the 1.6× in the comparison above comes from;
the other four operations moved by less than 5%.

## What shaped the kernels

Five compiler constraints are worth recording, because each is silent until the
Tile IR assembler or the compiler runs:

- **Tile dimensions are powers of two.** A row tile of 257 or 768 columns is
  rejected at Tile IR assembly with `failed to compile Tile IR program` and no
  further detail. LayerNorm therefore pads a row to the next power of two (768 →
  1024), and the kernel divides by the true width; the padded lanes are zeros,
  so they add nothing to either reduction. That costs 33% of the lanes on this
  shape and still lands level with the two-warp cuda-oxide kernel.
- **A partition load indexed by a loop variable does not vary.** The triangle
  kernel first walked the channel axis in a `for` loop, loading
  `a.partition([BI, N, 1]).load([pid.0, 0, channel])`; every channel received the
  first channel's value. Taking the channel from the third grid axis — the
  batched-GEMM idiom of the upstream `batch_matmul` example — fixes it.
- **A `Tile<..>` in an expression position is not rewritten** by the entry
  macro, so `None::<Tile<bool, { [] }>>` fails to resolve as a type while the
  same type in a `let` annotation is fine. The pointer loads carry their tuple
  type as an annotation and leave the mask and padding arguments unannotated.
- **`convert_scalar` has no `u32` → `i32`.** The CSR arrays are uploaded as
  `i32` and read through `*const i32`, which the host's bounds check (row
  pointers non-decreasing and ending at the edge count, every index naming a
  row) makes safe.
- **Scalar comparison is not a supported binary operator.** The edge walk is a
  counted range loop over `end - start` rather than `while edge < end`.

The irregular operation needed the raw-pointer path: the neighbor kernel reads
its row pointers, edge indices and weights with `load_ptr_tko` over device
pointers, and gathers the source row through a partition view indexed by the
loaded value. That is the distance between the safe tile model and the escape
hatch the plan anticipated, and it is why it was ported last.

## What the numbers do not establish

- No hardware counters were available on this host, so occupancy, register
  pressure and memory traffic are inferred from ratios, not read.
- LayerNorm's kernel time includes the 768 → 1024 row padding; a kernel with a
  native 768-wide tile would do less work.
- Event spans are sensitive to device contention. A repeat of this comparison
  that shared the GPU with another Nsight capture measured 3–20× larger spans for
  the same binary; the retained run is the one taken with the device idle, and it
  agrees with the earlier four-operation round to a few percent. A second attempt
  at 00:36 was discarded for the same reason, and the tell was not the tile
  column: PyTorch's own matmul kernel rose from 44.04 to 65.21 µs, which no change
  of mine can explain.
- One capture per operation; kernel time carries no interval. Event spans come
  from three rounds with rotating order; the per-round spread lives in the
  `results.json` the reproduction command below writes.
- Clocks are unlocked, and the WSL timestamp fallback used by Nsight Systems has
  reduced precision.
- Compilation, input transfers, process startup and end-to-end service work are
  excluded, as in the earlier rounds.
- The tile-sweep table was measured before the autotuner ran and is quoted from
  that session; every comparison and launch-path table is from one later session
  with the tuned tile.
- The cuda-oxide column is the third note's retained measurement, not taken in
  the same session as these numbers; the kernel track has since published mage-006
  (#58) with lower oxide numbers of its own (matrix multiply 68.62 µs of kernel
  time) measured against its own Triton and PyTorch controls. The two Rust columns
  should not be differenced across the two records.

## Open items

1. **Graphs for the other three kernels**: layer norm, triangle contraction and
   neighbor aggregation reject `--mode graph` today; each needs its own capture
   because a graph is recorded per launch shape.
2. **Tune the triangle and neighbor tiles**: neither was ever swept, and the
   neighbor kernel is the track's weakest result — 3.4× behind the hand-written
   kernel at 35.37 µs.
3. **Lower precision**: FP16/BF16/TF32 are separate contracts with their own
   error budgets; nothing here speaks to them.

## Reproduction

```bash
source scripts/cutile-env.sh
cd examples/cutile && cargo build --release && cd ../..
.venv/bin/python examples/oxide/comparison.py --implementation cutile \
  --experiment mage-004 --output artifacts/mage-004-comparison \
  --rounds 3 --iterations 100 --warmup 25
```

`--ops matmul` (or any subset) restricts a run. `CUTILE_MATMUL_TILE=BM,BN,BK`
selects a different matmul specialization, which is how the sweep above was
taken; `--mode single|batch|graph` selects the launch path; `--features tune`
adds the autotuner. Run one device experiment at a time: a concurrent capture
inflates these spans.
