# mage-004: cuTile Rust tile kernels

Status: **all five operations ported and measured** (2026-09-10).

This round adds a third Rust-to-CUDA path to the comparison. The first two are
in [mage-001](mage-001-comparison.md) (first cuda-oxide kernels) and
[mage-003](mage-003.md) (the same kernels after the shared-read and two-warp
revisions). cuTile Rust writes *tile* programs — single-threaded code over tiles
that the compiler maps onto warps, blocks and shared memory through CUDA Tile IR
— where cuda-oxide writes *thread* programs with the register tiles, shared
layouts and barriers chosen by hand.

The question is the one [kernel-exploration](../research/kernel-exploration.md)
asks: does layout and reuse decide the result more than the source language? The
three measured wins of mage-003 were all tiling decisions. Here the compiler
makes them, and the harness measures what that is worth.

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
| Matrix multiplication 1024³ | 52.99 | 91.92 | 218.50 | 82.6 |
| Bias + GELU 4096×768 | 36.01 | 27.21 | 30.84 | 13.3 |
| LayerNorm 4096×768 | 20.75 | 27.65 | 36.68 | 12.9 |
| Triangle contraction 128×32 | 61.11 | 107.83 | 165.22 | 83.5 |
| Neighbor aggregation 4096×64×65536 | 98.46 | 28.07 | 62.12 | 12.9 |

## GPU kernel time (µs per operation, single capture, 100 calls)

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust (mage-003) |
| --- | --- | --- | --- | --- |
| Matrix multiplication 1024³ | 44.04 | 83.61 | 172.52 | 80.00 |
| Bias + GELU 4096×768 | 16.12 | 7.76 | **7.97** | 11.0 |
| LayerNorm 4096×768 | 18.58 | 8.08 | 10.46 | 10.05 |
| Triangle contraction 128×32 | 28.49 | 93.90 | 120.72 | 80.1 |
| Neighbor aggregation 4096×64×65536 | 73.07 | 7.97 | 35.36 | 10.3 |

Kernel time is summed over the launches one operation needs — PyTorch's bias +
GELU is two kernels, its triangle contraction three and its neighbor
aggregation four; every other cell is one. The launch counts come from
`comparison-profiles.json`, not from a count of what the operation *should*
launch.

The two views disagree, and the disagreement is the result. On GPU time the tile
kernels are **ahead of cuda-oxide on bias + GELU** (7.97 against 11.0 µs) and
level with it on layer norm (10.46 against 10.05 µs), while trailing on the
other three: 2.2× on matrix multiply, 1.5× on triangle contraction and 3.4× on
neighbor aggregation. On event spans the tile kernels trail everywhere,
including where their kernels are faster: bias + GELU spans 30.84 µs around a
kernel that takes 7.97 µs.

That gap is the launch path, not the kernel. Each timed iteration here records
an event, launches, records a second event and synchronizes; for a runtime whose
device operations are lazy, that pattern serializes submission and measures it.
The cuda-oxide binary launches a driver kernel directly. **No claim about the
relative host cost of the two Rust runtimes follows from this table** — the
launch-path comparison in
[the research plan](../research/kernel-exploration.md) is what would separate
them, and cuTile Rust can express all three modes it asks for (single launch, a
batch divided by repetitions, and CUDA graph replay).

### The launch path, measured

The same launches, awaited one at a time versus queued ten at a time behind one
event pair (`--mode single` and `--mode batch --batch 10`), median µs per launch,
100 samples after 25 warmup launches, beside the kernel time from the capture
above:

| Operation | single | batch:10 | kernel |
| --- | --- | --- | --- |
| Matrix multiplication 1024³ | 192.45 | 160.46 | 172.52 |
| Bias + GELU 4096×768 | 24.64 | 8.40 | 7.97 |
| LayerNorm 4096×768 | 29.86 | 10.64 | 10.46 |
| Triangle contraction 128×32 | 135.07 | 118.61 | 120.72 |
| Neighbor aggregation 4096×64×65536 | 49.25 | 33.18 | 35.36 |

The difference between the first two columns is what the host spent per launch
while every launch was awaited: about 16 µs on the small operations and 32 µs on
matrix multiply. Batched, the spans converge on the kernel time — bias + GELU and
layer norm land within 0.5 µs of their kernels, and matrix multiply's batched
span sits *below* its kernel time because submission overlaps execution.

The cuda-oxide spans in the event table (13.3 µs for bias + GELU over an 11.0 µs
kernel) put that runtime's per-launch host cost at 2–3 µs. So on these shapes the
tile runtime's host path is roughly 7× the SIMT one's, and the event-span column
above is inflated by exactly that difference. That is the answer to the question
mage-002 left open, for the span view: the native advantage in the span column is
a launch-path artifact, not a kernel one.

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
| Matrix multiplication 1024³ | 188.42 | 175.67 | 183.69 | **161.77** | 182.10 | 172.52 |
| Bias + GELU 4096×768 | 24.91 | 8.91 | 9.82 | 8.67 | **7.73** | 7.97 |

Replay is the best mode for both: bias + GELU lands at 7.73 µs against a 7.97 µs
kernel, so the host cost that dominated the single-launch column is gone
entirely, and matrix multiply's replayed span sits below its kernel time because
capture removes per-launch submission from the critical path. Batching and
replay are within noise of each other on these shapes; what both establish is
that the span column's tile-runtime penalty is a submission artifact rather than
a property of the kernels.

### Matched launch paths, all three implementations

The comparison's span column awaits every call, for every implementation. Timing
PyTorch and Triton batched by ten the same way (`measure` in the harness's
`experiment.py` gains no argument for this; the numbers below come from a script
under `artifacts/cutile-dev/`), median µs per call, 100 samples after 25 warmup
calls:

| Operation | PyTorch single → batched | Triton single → batched | cuTile single → batched |
| --- | --- | --- | --- |
| Matrix multiplication 1024³ | 47.10 → 46.29 | 87.04 → 75.16 | 185.34 → 170.39 |
| Bias + GELU 4096×768 | 18.43 → 19.75 | 19.46 → 13.32 | 24.58 → 8.47 |
| LayerNorm 4096×768 | 13.31 → 12.07 | 20.48 → 13.72 | 29.66 → 10.63 |
| Triangle contraction 128×32 | 46.80 → 35.50 | 88.38 → 77.72 | 132.29 → 120.00 |
| Neighbor aggregation 4096×64×65536 | 79.87 → 67.88 | 20.48 → 13.82 | 48.99 → 33.18 |

Batching helps every implementation, and helps the tile runtime most because its
per-call host cost was the largest. With the launch path matched, bias + GELU
(8.47) and layer norm (10.63) become the tile kernels' wins, matrix multiply and
triangle contraction stay behind because those kernels are genuinely slower
(172.52 and 120.72 against 44.04 and 28.49), and neighbor aggregation lands
between PyTorch and Triton. Triton's own launcher costs 6–7 µs per awaited call
(19.46 → 13.32 on bias + GELU); PyTorch's costs about 1 µs.

## Correctness

Every value in both tables passed the full-element check on every round. Worst
observed error:

| Operation | max abs error | tolerance |
| --- | --- | --- |
| Matrix multiplication | 1.48e-05 | 1e-4 |
| Bias + GELU | 2.38e-07 | 1e-4 |
| LayerNorm | 4.77e-07 | 1e-4 |
| Triangle contraction | 7.15e-07 | 1e-4 |
| Neighbor aggregation | 3.58e-07 | 1e-4 |

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

The retained tile is 128 × 64 × 8, 3.7× faster than the starting one. The shape
is not monotone in any single dimension: deepening K from 8 to 32 costs 42%
at 16 × 16 but only 6% at 64 × 64, and 128 × 128 × 8 is 2.3× slower than
128 × 64 × 8, which is the signature of a register or occupancy cliff rather
than of arithmetic. This is the same class of result the thread-level kernels
reported in mage-003: the ratio between two builds is what identifies the limit
when hardware counters are unavailable.

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
  that shared the GPU with another Nsight capture measured 3–20× larger spans
  for the same binary; the retained run is the one taken with the device idle,
  and it agrees with the earlier four-operation round to a few percent.
- One capture per operation; kernel time carries no interval. Event spans come
  from three rounds with rotating order; the per-round spread lives in the
  `results.json` the reproduction command below writes.
- Clocks are unlocked, and the WSL timestamp fallback used by Nsight Systems has
  reduced precision.
- Compilation, input transfers, process startup and end-to-end service work are
  excluded, as in the earlier rounds.

## Open items

1. **Graphs for the other three kernels**: layer norm, triangle contraction and
   neighbor aggregation reject `--mode graph` today; each needs its own capture
   because a graph is recorded per launch shape.
3. **Tuning**: the twelve-configuration sweep above is bounded and hand-picked.
   `cutile::tune` ships an experimental autotuner that would search it properly,
   and the same question applies to the triangle and neighbor tiles, which were
   never swept.
4. **Lower precision**: FP16/BF16/TF32 are separate contracts with their own
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
taken. Run one device experiment at a time: a concurrent capture inflates these
spans.
