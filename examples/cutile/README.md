# Mage / cuTile Rust experiments

Tile-based FP32 kernels in [cuTile Rust](https://github.com/NVlabs/cutile-rs),
built and measured through the same harness as `examples/oxide`.

- [Setup and build](https://github.com/superposition/mage/blob/master/docs/guide.md)
- [The cuTile track plan](https://github.com/superposition/mage/blob/master/docs/research/cutile-rust.md)
- [mage-004: the measured round](https://github.com/superposition/mage/blob/master/docs/experiments/mage-004.md)
- [Profiling](https://github.com/superposition/mage/blob/master/docs/profiling.md)

The binary reads the same input directory as `examples/oxide`:
`input.json` (`op`, `dims`, `warmup`, `iterations`) plus raw little-endian
files — `a.bin`, `b.bin`, and LayerNorm's `c.bin`; CSR uses unsigned 32-bit
`rowptr.bin` and `indices.bin` with weights in `b.bin`. `experiment.py`
generates those files and hashes them, and `rust-output.bin` is what it checks
against PyTorch.

```bash
source scripts/cutile-env.sh          # CUDA 13.3 toolkit + tileiras
cd examples/cutile && cargo build --release
cd ../..
.venv/bin/python examples/oxide/comparison.py --implementation cutile \
  --experiment mage-004 --output artifacts/mage-004-comparison \
  --rounds 3 --iterations 100 --warmup 25
```

`--capture` brackets the measured region with the CUDA profiler API so an
Nsight Systems capture contains only the timed launches; `--iterations N`
overrides the manifest count. `--mode single` (the default) awaits every launch
and brackets it with one event pair; `--mode batch --batch N` queues N launches
behind one event pair and divides, which separates the kernel from the host
submission path. Each run retains its output and event samples in
`rust-runs/<run-id>/`, with root files mirroring the latest run, and the timing
record names the mode it was taken with.

## Kernels

| Operation | Kernel | Notes |
| --- | --- | --- |
| matmul | `matmul` | `[BM, BN]` output tile per program, `BM × BN × BK = 128 × 64 × 8` from the mage-004 sweep |
| bias + GELU | `bias_gelu` | `[8, 128]` tiles, bias taken from the tile's column block |
| layernorm | `layer_norm` | one row per program; the row is padded to the next power of two because tile dimensions must be, and the kernel divides by the true width |
| triangle contraction | `triangle` | channel axis as the third grid axis, one `mma` per program |
| neighbor aggregation | `neighbor` | one row per program, CSR read through raw pointers (`load_ptr_tko`): the edge walk bound is data, not a partition shape |

Five constraints of the toolchain shaped these kernels. Each is silent until the
assembler or the compiler runs, and each is recorded next to the code that hit
it:

- **Tile dimensions must be powers of two.** A 257- or 768-wide row tile is
  rejected with `failed to compile Tile IR program` and no further detail.
- **A partition load indexed by a loop variable does not vary.** Take a varying
  index from the grid axes instead: the triangle kernel did not until it did.
- **A `Tile<..>` in an expression position is not rewritten** by the entry
  macro, so `None::<Tile<bool, { [] }>>` will not resolve.
- **`convert_scalar` has no `u32` → `i32`**; the CSR arrays are uploaded as
  `i32`.
- **Scalar comparison is not a supported binary operator**; count the edges and
  loop over the count.

`CUTILE_MATMUL_TILE=BM,BN,BK` selects a different matmul specialization; that is
how the mage-004 tile sweep was taken. Tile shapes are part of the JIT
specialization key, so each new shape compiles once at first launch, outside the
timed region. The padded shape a run actually executed is recorded in
`rust-timing.json`.

Licensed Apache-2.0, like the upstream project it builds on. See `LICENSE`.
