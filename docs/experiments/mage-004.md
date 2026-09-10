# mage-004: cuTile Rust tile kernels

Status: **all five operations ported and measured** (2026-09-10), except where
noted for neighbor's kernel time.

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
| Matrix multiplication 1024³ | 58.50 | 93.88 | 216.64 | 82.6 |
| Bias + GELU 4096×768 | 35.37 | 26.95 | 30.22 | 13.3 |
| LayerNorm 4096×768 | 23.46 | 26.69 | 34.21 | 12.9 |
| Triangle contraction 128×32 | 61.42 | 102.09 | 153.71 | 83.5 |
| Neighbor aggregation 4096×64×65536 | 94.45 | 26.54 | 63.96 | 12.9 |

## GPU kernel time (µs per launch, single capture, 100 launches)

| Operation | PyTorch | Triton | cuTile Rust | cuda-oxide Rust (mage-003) |
| --- | --- | --- | --- | --- |
| Matrix multiplication 1024³ | 44.0 | 71.9 | 176.27 | 80.00 |
| Bias + GELU 4096×768 | 15.8 | 7.6 | **8.09** | 11.0 |
| LayerNorm 4096×768 | 11.2 | 8.0 | 10.66 | 10.05 |
| Triangle contraction 128×32 | 28.5 | 81.6 | 120.58 | 80.1 |
| Neighbor aggregation | 67.3 | 8.0 | not captured | 10.3 |

The two views disagree, and the disagreement is the result. On GPU time the tile
kernels are **ahead of cuda-oxide on bias + GELU** (8.09 against 11.0 µs) and
level with it on layer norm (10.66 against 10.05 µs), while trailing on the two
matmul-shaped operations (2.2× and 1.5×). On event spans the tile kernels trail
everywhere, including where their kernels are faster: bias + GELU spans 30.22 µs
around a kernel that takes 8.09 µs.

That gap is the launch path, not the kernel. Each timed iteration here records
an event, launches, records a second event and synchronizes; for a runtime whose
device operations are lazy, that pattern serializes submission and measures it.
The cuda-oxide binary launches a driver kernel directly. **No claim about the
relative host cost of the two Rust runtimes follows from this table** — the
launch-path comparison in
[the research plan](../research/kernel-exploration.md) is what would separate
them, and cuTile Rust can express all three modes it asks for (single launch, a
batch divided by repetitions, and CUDA graph replay).

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
- Neighbor's kernel time has no capture yet: the device was busy with another
  agent's profiling run when this round closed. Its event span is retained.
- Event spans are sensitive to device contention. A repeat of this comparison
  that shared the GPU with another Nsight capture measured 3–20× larger spans
  for the same binary; the retained run is the one taken with the device idle,
  and it agrees with the earlier four-operation round to a few percent.
- One capture per operation; kernel time carries no interval. Event spans come
  from three rounds with rotating order, and the round means above are quoted
  with their spread in `artifacts/mage-004-comparison/results.json`.
- Clocks are unlocked, and the WSL timestamp fallback used by Nsight Systems has
  reduced precision.
- Compilation, input transfers, process startup and end-to-end service work are
  excluded, as in the earlier rounds.

## Open items

1. **The launch path**: async device operations, batching and CUDA graph replay,
   with warmup excluded. Until then the event-span column cannot separate a
   slower runtime from a slower kernel.
2. **Tuning**: the twelve-configuration sweep above is bounded and hand-picked.
   `cutile::tune` ships an experimental autotuner that would search it properly,
   and the same question applies to the triangle and neighbor tiles, which were
   never swept.
3. **Neighbor's kernel time**, once the device is free.
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
