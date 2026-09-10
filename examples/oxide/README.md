# Mage / cuda-oxide experiments

Five forward FP32 kernels in Rust and Triton, with a shared-input PyTorch reference harness.

- [Setup and build](https://github.com/superposition/mage/blob/master/docs/guide.md)
- [Ideas and profile graphs](https://superposition.github.io/mage/experiments/mage-001/#profiles)
- [Contracts](https://github.com/superposition/mage/blob/master/docs/kernel-contracts.md)
- [Three-implementation comparison and reproduction](https://github.com/superposition/mage/blob/master/docs/experiments/mage-001-comparison.md)
- [Python and native profiling](https://github.com/superposition/mage/blob/master/docs/profiling.md)

The Rust executable reads `INPUT_DIR/input.json` (`op`, `dims`, `warmup`,
`iterations`) and raw little-endian files. Floating-point inputs are `a.bin`,
`b.bin`, and LayerNorm's `c.bin`. CSR uses unsigned 32-bit `rowptr.bin` and
`indices.bin`; weights are `b.bin`. The harness generates these files and hashes.

`--capture` brackets the measured region with CUDA profiler APIs.
`--iterations N` overrides the manifest count. Each run retains output and
timing in `rust-runs/<run-id>/`, with root files mirroring the latest run.
Allocation, transfers, and warmup precede capture.

The native host validates dimensions, file sizes, finite inputs, and CSR bounds.
`experiment.py` checks every output against PyTorch with TF32 disabled.
`profile_suite.py` captures both languages serially through Mage.
Pass `--languages python triton rust` to include the fixed-tile Triton examples.
`comparison.py` rotates the measurement order across three rounds and retains
all timing samples and full-output checks. These examples are forward-only
learning kernels; they do not replace Mage's existing general operators.

The Rust example is Apache-2.0 licensed and adapts the shared-memory matmul
structure from NVIDIA's cuda-oxide `tiled_gemm` example. See `LICENSE` and
the attribution at the start of `src/main.rs`.
