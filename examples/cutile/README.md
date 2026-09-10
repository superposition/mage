# Mage / cuTile Rust experiments

Tile-based FP32 kernels in [cuTile Rust](https://github.com/NVlabs/cutile-rs),
built and measured through the same harness as `examples/oxide`.

- [Setup and build](https://github.com/superposition/mage/blob/master/docs/guide.md)
- [The cuTile track plan](https://github.com/superposition/mage/blob/master/docs/research/cutile-rust.md)
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
.venv/bin/python examples/oxide/experiment.py \
  --binary examples/cutile/target/release/mage-cutile --output artifacts/mage-004
```

`--capture` brackets the measured region with the CUDA profiler API so an
Nsight Systems capture contains only the timed launches; `--iterations N`
overrides the manifest count. Each run retains its output and event samples in
`rust-runs/<run-id>/`, with root files mirroring the latest run.

## Kernels

| Operation | Kernel | Notes |
| --- | --- | --- |
| matmul | `matmul` | [BM, BN] output tile per program, strict FP32 accumulation |
| bias + GELU, layer norm, triangle contraction, neighbor aggregation | — | not ported yet; `examples/oxide` covers all five |

Tile shapes are compile-time values and part of the JIT specialization key, so
the harness keeps one shape per run and records the padded shape in
`rust-timing.json`. The 16 × 16 × 8 tiles are a starting point, not a tuned
choice.

The cuTile compiler lowers these kernels through CUDA Tile IR; the first launch
of a specialization compiles it, so every launch in the warmup and measured
regions runs a cached cubin.

Licensed Apache-2.0, like the upstream project it builds on. See `LICENSE`.
