# Mage / cuda-oxide experiments

Five forward FP32 kernels and a shared-input PyTorch reference harness.

- [Setup and build](https://github.com/superposition/mage/blob/master/docs/guide.md)
- [Contracts and measured evidence](https://superposition.github.io/mage/experiments/mage-001/)
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

The Rust example is Apache-2.0 licensed and adapts the shared-memory matmul
structure from NVIDIA's cuda-oxide `tiled_gemm` example. See `LICENSE` and
the attribution at the start of `src/main.rs`.
