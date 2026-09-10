# Adding a cuTile Rust track

Status: **all five operations are ported and measured** (2026-09-10), with
neighbor's kernel time still to capture. The toolkit gate passed,
`examples/cutile` builds and runs, and [mage-004](../experiments/mage-004.md)
holds the values: the tile kernels are ahead of cuda-oxide on GPU time for
bias + GELU, level on layer norm, 2.2× behind on matrix multiply and 1.5× behind
on triangle contraction, with the retained matmul tile chosen by a
twelve-configuration sweep. The launch path, autotuning and the lower-precision
contracts remain open; see [Open work](#open-work).

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

Only the toolkit was missing. It is now in place at `/usr/local/cuda-13.3`,
installed on 2026-09-10 **without touching the package database**: an earlier
NVIDIA DKMS build left `linux-headers-5.15.0-191-generic` half-configured, apt
refuses to run in that state, and the toolkit pieces cuTile actually reads total
about 90 MB. Each one was fetched from the configured NVIDIA repository and
extracted with `dpkg-deb -x /`:

| Piece | Package | Why it is needed |
| --- | --- | --- |
| `bin/tileiras` 13.3.36 | `cuda-tileiras-13-3` | the Tile IR assembler `cutile-compiler` invokes; ships with 13.2+ |
| `version.json` 13.3.1 | `cuda-toolkit-13-3` | toolkit version metadata |
| `bin/nvcc`, `crt/`, `include/crt` | `cuda-nvcc-13-3`, `cuda-crt-13-3` | compiler the host crates probe |
| `include/cuda.h`, `lib64/libcudart*` | `cuda-driver-dev-13-3`, `cuda-cudart{,-dev}-13-3` | `cuda-bindings/wrapper.h` and the runtime |
| `include/curand.h` | `libcurand-dev-13-3` | second include in `wrapper.h`; bindgen fails without it |
| `nvvm/lib64/libnvvm.so.4` | `libnvvm-13-3` | `tileiras` `dlopen`s it; without it every compile fails as `failed to compile Tile IR program` |

Nothing else moved: `/usr/local/cuda` still points at `cuda-12.8`, the 13.0 tree
the oxide track pins is untouched, and no package was installed, upgraded or
removed. Restoring the previous state is `rm -rf /usr/local/cuda-13.3`; the same
tree can be replaced later by `apt-get install cuda-toolkit-13-3` once dpkg is
repaired.

## Layout in this worktree

| Path | Purpose |
| --- | --- |
| `scripts/cutile-env.sh` | `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3`, `CUTILE_TILEIRAS_PATH`, `PATH`. Never sourced in the same shell as `scripts/oxide-env.sh`, which pins 13.0. |
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

### Phase 0 — toolkit and smoke test (the gate) — **passed 2026-09-10**

```bash
source scripts/cutile-env.sh
cd ~/code/cutile-rs   # upstream v0.3.1 clone used for the smoke test
for ex in hello_world saxpy gemm rms_norm softmax; do
  cargo +1.98.1 run --quiet -p cutile-examples --example "$ex"
done
```

Result on the RTX 4090 through WSL2:

| Example | Outcome |
| --- | --- |
| `hello_world` | `Hello, I am program <0, 0, 0> in a kernel with <1, 1, 1> programs.` |
| `saxpy`, `gemm`, `rms_norm`, `softmax` | exit 0, printed values match their checks (`gemm` 8192, `softmax` rows sum to 1) |

Tile IR does target sm_89, so the track is live and no "unavailable" note is
needed. Two prerequisites surfaced during the build, both now in place:

- **`libnvvm.so`.** `tileiras` accepts the bytecode, writes the object, then
  fails with only `error: failed to compile Tile IR program` (exit 5) because it
  `dlopen`s `$CUDA_TOOLKIT_PATH/nvvm/lib64/libnvvm.so` and that directory was
  absent from the minimal extraction. Found with
  `strace -f -e trace=openat tileiras ... | grep ENOENT`. `libnvvm-13-3` fixes it.
- **`curand.h`.** `cuda-bindings`'s build script runs bindgen over
  `wrapper.h`, which includes `<cuda.h>` and `<curand.h>`; `libcurand-dev-13-3`
  supplies the second.

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
| ~~sm_89 is not a Tile IR target~~ | Phase 0 `hello_world`, `saxpy`, `gemm`, `rms_norm`, `softmax` | **Resolved:** all five run on the 4090. |
| ~~Driver 591.74 rejects the emitted bytecode version~~ | Phase 0 compile path | **Resolved:** `tileiras --list-versions` reports 13.1, 13.2, 13.3 and compiles sm_89 cubins. |
| JIT compile time lands inside the capture | Timing record and nsys trace | Warm every specialization before capture; the harness already separates warmup. |
| Tile shape is part of the specialization, so a sweep recompiles | Wall clock of Phase 1 | Keep shapes fixed as the earlier rounds do; use `-1` dimensions where the contract allows. |
| The tile model cannot express CSR gather | Phase 2 | `*_tko` raw pointers, measured against the safe path; report the cost. |
| The results differ from oxide by less than the harness resolves | Phase 3 | The difference is the finding; both binaries are retained. |

## Open work

| Item | State | Next test |
| --- | --- | --- |
| The launch path | Not measured: the harness serializes submission per iteration, which prices the lazy runtime's host work rather than its queueing | Single launch, a batch divided by repetitions, and CUDA graph replay, with warmup excluded |
| Autotuning | Not used: the retained matmul tile came from twelve hand-picked configurations | `cutile::tune` over the same space, with the warm-up outside the timed region; the triangle and neighbor tiles were never swept at all |
| Neighbor kernel time | Event span retained, capture missing: the device was busy with another agent's profiling run | One Nsight Systems capture of 100 launches, device idle |
| Lower precision | Not measured | FP16/BF16/TF32 as separate contracts with their own error budgets |

## Where the results live

- [mage-004](../experiments/mage-004.md) — method, event spans, kernel times,
  the tile sweep, and what the numbers do not establish.
- `docs/assets/results/mage-004/` — the portable evidence the figures are drawn
  from: `comparison-results.json` (every sample and every full-element check)
  and `comparison-profiles.json` (the per-launch kernel durations). Both are
  written by `examples/oxide/export_comparison.py`, which the figure script
  below drives.
- `artifacts/cutile-dev/figures.sh` — captures, exports and plots in one run.
  It needs the device to itself, so run it when no other agent is profiling.

```bash
source scripts/cutile-env.sh
cd examples/cutile && cargo build --release && cd ../..
.venv/bin/python examples/oxide/comparison.py --implementation cutile \
  --experiment mage-004 --output artifacts/mage-004-comparison \
  --rounds 3 --iterations 100 --warmup 25
bash artifacts/cutile-dev/figures.sh   # captures, export, figures
```
