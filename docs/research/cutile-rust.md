# Adding a cuTile Rust track

Status: **planned; the toolkit prerequisite is the only thing missing** (2026-09-10).

[cuTile Rust](https://github.com/NVlabs/cutile-rs) is NVlabs' second Rust-to-CUDA
stack, alongside [cuda-oxide](https://github.com/NVlabs/cuda-oxide). Where
`examples/oxide` writes SIMT kernels — threads, shared-memory layouts, barriers
chosen by hand — cuTile Rust writes single-threaded programs over tiles and lets
the compiler map them onto warps, blocks, Tensor Cores and TMA through CUDA Tile
IR. This document plans the same five FP32 operations as a third native
implementation, measured with the existing harness, so the two Rust models and
Triton can be compared on one contract.

The work happens in the worktree `~/code/mage-cutile` on branch
`feat/cutile-rust`, based on `origin/master` (`8c7c097`). `~/code/mage-oxide`
stays on `feat/evolution-loop`; the two trees are identical today (`git diff
--stat origin/master feat/evolution-loop` is empty), so neither track blocks the
other.

## Why this is worth measuring

- **The three measured wins in mage-003 were tiling decisions.** Reading shared
  tiles as 128-bit quads, transposing the `A` tile for contiguous register
  loads, and deepening the K step moved matrix multiply from 141.70 to 80.00 µs;
  two warps per row moved layer norm from 11.03 to 10.05 µs. cuTile Rust makes
  those decisions inside the compiler. Either it makes them for free, or the
  hand-written SIMT kernels keep an advantage — both outcomes answer the
  open question in [kernel-exploration.md](kernel-exploration.md) about whether
  layout and reuse matter more than the source language.
- **The launch path is first-class.** cuTile device ops are lazy, support
  asynchronous execution and CUDA graph replay without extra glue. The pending
  mage-002 question — does the native advantage survive a matched launch path —
  can be asked of a runtime that expresses all three timing modes natively.
- **Autotuning ships with the library** (`cutile::tune`, experimental), which is
  the bounded tile-shape sweep the research plan asks for.
- **Ownership at the launch boundary** is the library's safety claim (mutable
  tensors partitioned disjointly, immutable ones shared, launchers that preserve
  ownership while work is in flight). The measurable question is what that
  discipline costs on five fixed shapes.

## What this host has, and what it lacks

| Fact | Value | Where it comes from |
| --- | --- | --- |
| GPU | RTX 4090, compute capability 8.9, 24 GB, driver 591.74 | `nvidia-smi` |
| Installed toolkits | `cuda-toolkit-13-0` (13.0.3-1) and 12.8; `/usr/local/cuda` → `cuda-12.8` | `dpkg -l`, `ls -l /usr/local/cuda` |
| cuTile toolkit floor | CUDA **13.2**; `tileiras` ships with 13.2 | `cutile-compiler/src/cuda_tile_runtime_utils.rs`: `MIN_TILE_CUDA_VERSION = 13020` |
| Architecture support | `sm_80`+; `sm_8x` added in CUDA 13.2; 13.3 recommended | cutile-rs README |
| `tileiras` present? | No — absent from both 13.0 and 12.8 | `ls /usr/local/cuda-13.0/bin/tileiras` |
| CUDA 13.3 packages | `cuda-toolkit-13-3` 13.3.1-1 and `cuda-tileiras-13-3` 13.3.36-1 in the configured NVIDIA repo | `apt-cache madison` |
| Install footprint | 62 packages, no driver package, 0 removed | `apt-get install -s --no-install-recommends` |
| Root access | Available without a password via `wsl -d Ubuntu-22.04 -u root` | `id` → `uid=0(root)` |
| Rust | default 1.85.0, cuTile MSRV **1.89**, stable 1.98.1 available | `rustc --version`, `rustup check`, workspace `Cargo.toml` |
| Rust nightly (oxide track) | `nightly-2026-08-28` + `cargo-oxide` @ `26754ae` | [guide.md](../guide.md) |
| Upstream clone | `~/code/cutile-rs` @ `v0.3.1` = `cdc69c13a7529552a26d9941893f047970d8e95f` | `git describe` |
| crates.io | `cutile = "0.3.1"` is published | `cargo search cutile` |

Only the toolkit install is missing.

## Layout in this worktree

| Path | Purpose |
| --- | --- |
| `scripts/bootstrap-cutile.sh` | Install `cuda-toolkit-13-3` with `--no-install-recommends`, verify `tileiras`, and leave `/usr/local/cuda` pointing where it pointed before. |
| `scripts/cutile-env.sh` | `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3` and `PATH`. Never sourced in the same shell as `scripts/oxide-env.sh`, which pins 13.0. |
| `examples/cutile/` | Crate `mage-cutile`: the five kernels, the same CLI and file contract as `examples/oxide`. |
| `examples/cutile/rust-toolchain.toml` | Pin the stable channel (1.98.1 today), unlike the oxide track's pinned nightly. |
| `docs/experiments/mage-004.md` | Measurement record: method, retained values, and the variants that measured worse. |
| `docs/assets/figures/mage-004/` | Figures produced by the existing plot scripts. |

Out of scope for the first pass: replacing Mage's Triton operators, autograd, or
the Python API. Like the oxide examples, these are forward-only learning
kernels.

## The contract the cuTile binary must meet

`examples/cutile` is a drop-in peer of `examples/oxide`, so that one harness can
drive either:

- reads `INPUT_DIR/input.json` (`op`, `dims`, `warmup`, `iterations`) and the raw
  little-endian files (`a.bin`, `b.bin`, `c.bin`; CSR `rowptr.bin`,
  `indices.bin`);
- accepts `--iterations N` and `--capture` (CUDA profiler start/stop through
  `libcuda.so.1`, as `examples/oxide/src/main.rs:530` does with `libloading`);
- retains output and event samples in `rust-runs/<run-id>/`, mirroring the
  latest run at the input-directory root;
- validates dimensions, file sizes, finite inputs and CSR bounds before
  launching; FP32 arithmetic with no tensor cores, checked against PyTorch with
  TF32 disabled.

Same bytes in, same JSON out is what lets `comparison.py` rotate
implementations and `profile_suite.py` capture both in one session. `mage
profile-exec` already accepts any executable
(`src/mage/cli.py:371`, `src/mage/profiler/backends/native.py:44`), so profiling
a cuTile binary needs no change to the `mage` package.

## Phases

### Phase 0 — toolkit and smoke test (the gate)

```bash
# one-time, as root through WSL; no driver package is touched
wsl -d Ubuntu-22.04 -u root bash -lc \
  'DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-13-3'
ls /usr/local/cuda-13.3/bin/tileiras

# upstream smoke test, from ~/code/cutile-rs
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo +1.98.1 run -p cutile-examples --example hello_world
cargo +1.98.1 run -p cutile-examples --example saxpy
cargo +1.98.1 run -p cutile-examples --example gemm
```

Gate: `hello_world` prints its tile line on the 4090 and `gemm` produces a
correct result. If Tile IR cannot target sm_89, stop and report the track as
unavailable with the exact error — [kernel-exploration.md](kernel-exploration.md)
treats an unavailable mode as a result to report, not a workaround to invent.
Do not build the `cuda-tile-rs` member: it is excluded from default-members
because its build downloads and compiles LLVM.

### Phase 1 — one operation end to end

Matrix multiply first: highest signal (80.0 µs for oxide against 44.0 µs for the
cuBLAS call behind PyTorch), a tile kernel with an obvious partition, and
upstream references (`gemm`, `gemm_static`, `persistent_gemm`, `mxfp8`). Port the
input/output plumbing from `examples/oxide/src/main.rs` and produce a
`mage-cutile` binary that passes `experiment.py`'s full-output check and writes
the same timing record.

Gate: correctness passes, and an nsys capture through `mage profile-exec` yields
per-launch kernel durations.

### Phase 2 — the five operations

`matmul`, `bias + GELU`, `layer_norm`, triangle contraction, neighbor
aggregation. The first three are regular and map to tiles directly (upstream has
`rms_norm`, `softmax` and the pointwise kernels under `cutile-kernels/src`). The
last two are irregular — CSR gather and degree-skewed work. If the tile model
cannot express them, cuTile's raw-pointer escape hatches (`load_ptr_tko`,
`store_ptr_tko`, `unsafe`) are the fallback, and the distance between the safe
path and the escape hatch becomes part of the result rather than a footnote.

### Phase 3 — the comparison

Generalize the harness so both binaries are first-class:

- `examples/oxide/comparison.py:86` and `examples/oxide/profile_suite.py:31` take
  the binary from an `--implementation {oxide,cutile}` argument instead of the
  hard-coded `target/release/mage-oxide`;
- add a `mage-004` caption to the experiment table in
  `scripts/plot-comparison.py` (around line 32);
- keep `experiment.py --binary`, which already accepts an override.

Then three rotating rounds of 100 event samples per implementation per
operation, one Nsight Systems capture per case, a fresh-run confirmation, and
the `mage-004` record with the same three tables the earlier field notes use:
kernel time, event span, and the attempts that measured worse.

### Phase 4 — the launch-path question

cuTile device ops are lazy and support asynchronous execution and CUDA graph
replay, so the mage-002 question can be answered inside one language: single
launch, a batch of launches timed together and divided by repetitions, and graph
replay — over the same five operations, with warmup outside the timed region and
the same stream, synchronization and allocation policy in each mode.

### Phase 5 — write-up

`docs/experiments/mage-004.md` for method and values, a field note under
`docs/`, and the journal entry in the established form: the question, the
prediction that chose each change, and the attempts that measured worse. Then
`docs/guide.md` and the README's Rust section gain the cuTile build and
profiling commands next to the oxide ones.

## Risks, and the test that decides each

| Risk | Deciding test | If it fails |
| --- | --- | --- |
| sm_89 is not a Tile IR target for these kernels | Phase 0 `hello_world`, `gemm` | Keep the exact error, publish the track as unavailable, keep the oxide numbers. |
| Driver 591.74 rejects the emitted bytecode version | Phase 0 | Pin `CUTILE_BYTECODE_VERSION=13.2`, or install the 13.2 toolkit next to 13.3. |
| JIT compile time lands inside the capture | Timing record and nsys trace | Warm every specialization before capture; the harness already separates warmup. |
| Tile shape is part of the specialization, so a sweep recompiles | Wall clock of Phase 1 | Keep shapes fixed as the earlier rounds do; use `-1` dimensions where the contract allows. |
| The tile model cannot express CSR gather | Phase 2 | `*_tko` raw pointers, measured against the safe path; report the cost. |
| The results differ from oxide by less than the harness resolves | Phase 3 | The difference is the finding; both binaries are retained. |

## First commands once the plan is approved

```bash
wsl -d Ubuntu-22.04 -u root bash -lc \
  'DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-13-3'
wsl -d Ubuntu-22.04 -lc 'rustup toolchain install 1.98.1 --profile minimal'
wsl -d Ubuntu-22.04 --cd ~/code/cutile-rs -lc \
  'CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 cargo +1.98.1 run -p cutile-examples --example hello_world'
```
