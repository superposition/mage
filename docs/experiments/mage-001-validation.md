# mage-001: measurements and validation

[Experiment contract](https://superposition.github.io/mage/experiments/mage-001/) ·
[Journal](https://superposition.github.io/journal/why-this-notebook/)

## Completed checks

- All five default-size FP32 operations passed a full-element comparison against PyTorch with TF32 disabled.
- All ten boundary cases passed, including tile tails, constant LayerNorm, a single feature, empty adjacency, and duplicate edges.
- Python regression checks: **47 passed, 2 skipped**. These include native input rejection, profiler parsing, old-database migration, existing activations, and restoration of Triton hooks after repeated contexts and exceptions.
- Compute Sanitizer's memory checks for all five kernels reported **0 errors**; LayerNorm's race check reported **0 hazards**.
- Both Jekyll sites build with strict front matter. The technical page's equations render through MathJax.

## First event measurements

Measured on NVIDIA GeForce RTX 4090 using PyTorch 2.9.1+cu128,
Python 3.13.2, and the pinned Rust/CUDA build.
Each default case uses 10 warmup iterations followed by 100 samples.

| Operation | Rust event mean (µs) | PyTorch event mean (µs) | Maximum absolute error |
| --- | ---: | ---: | ---: |
| matmul | 345.37 | 56.00 | 0 |
| gelu | 12.75 | 21.22 | 5.31e-06 |
| layernorm | 24.70 | 17.71 | 4.77e-07 |
| triangle | 82.85 | 57.17 | 0 |
| neighbor | 12.82 | 105.12 | 2.98e-07 |

These are CUDA-event spans for the harness, with inputs resident on the GPU.
They exclude compilation and transfers, but short operations can include gaps
caused by host submission. The Python reference can launch multiple kernels;
the Rust examples each launch one. GPU clocks were not locked, and this is a
single WSL workstation session. Treat these as a starting observation, not a
general comparison of Python and Rust or a controlled speedup claim.

Download [all event samples, input hashes, source hashes, and environment metadata](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-001/event-results.json)
and [the boundary-case results](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-001/boundary-results.json).

## Systems captures

The five Rust captures each contain exactly **100 launches** after warmup.
The corresponding Python captures contain:

| Operation | Python launches for 100 iterations | Rust launches |
| --- | ---: | ---: |
| Matmul | 100 | 100 |
| Bias + GELU | 200 | 100 |
| LayerNorm | 100 | 100 |
| Triangle contraction | 300 | 100 |
| Neighbor aggregation | 400 | 100 |

These counts expose a useful distinction: an operation in the mathematical
description can expand into several GPU kernels. The profiles include names,
launch dimensions, durations, registers, and shared-memory allocation where
available. All ten sessions were saved to a separate SQLite database.

Download the [capture inventory](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-001/nsys-summary.json)
and [all 1,600 normalized per-launch rows](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-001/nsys-kernels.csv).
The inventory points to the retained local `.nsys-rep`, SQLite, log, and JSON/CSV
files. Large native reports are not embedded in this site.

## A failure that changed the setup

The first Systems captures contained CUDA API events but no GPU kernel rows.
NVIDIA's [2025.3 release notes](https://docs.nvidia.com/nsight-systems/2025.3/ReleaseNotes/index.html)
describe unreliable timestamp conversion on WSL and prescribe
`CuptiUseRawGpuTimestamps=false`. Applying that setting restored the kernel
traces. The reproduction guide now includes a script that updates only this option.
NVIDIA describes the fallback as less precise; keep that limitation beside
the Systems durations and use the separate event measurements for this table.

Nsight Compute currently reports **ERR_NVGPUCTRPERM**, including when invoked
as WSL root. Its parser and failure handling pass regression tests; hardware
counter capture on this host still requires the Windows NVIDIA counter-access
setting described in the [profiling guide](https://github.com/superposition/mage/blob/master/docs/profiling.md).
No occupancy, cache, or bandwidth result is claimed from that failed capture.

## What the evidence suggests next

The scalar tiled matmul leaves substantial room for optimization. The fusion
and aggregation examples also show why it is useful to count launches before
interpreting an event span. A next experiment can change the implementation,
hold the mathematical contract fixed, and examine the trace before drawing a
conclusion about where the time went.
