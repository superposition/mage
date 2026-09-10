# mage-001: PyTorch, Triton, and Rust comparison

[Public graphs and interpretation](https://superposition.github.io/mage/experiments/mage-001/#profiles)
· [Mathematical contracts](../kernel-contracts.md)

## What was measured

RTX 4090, WSL2, PyTorch 2.9.1+cu128, Triton 3.5.1, and the existing pinned
cuda-oxide build targeting sm_89. The five default shapes and input bytes are
identical across implementations. All use FP32; PyTorch TF32 is disabled and
Triton's dot products explicitly use `input_precision="ieee"`.

The Triton implementations are new, fixed-tile, forward-only counterparts in
`examples/oxide/triton_target.py`. They are **not Mage's autotuned general
operators**, `torch.compile` output, or tuned production kernels. Matmul and
triangle use 32×32×32 tiles; GELU uses 256 elements per program; LayerNorm
reduces one padded row; neighbor aggregation reduces chunks of 32 edges and
64 features. The Rust kernels are the unchanged first implementations.

### CUDA events

Three rounds of 100 measurements per implementation and operation, with 25
warmup calls per round. Implementation order rotates Python/Triton/Rust,
Triton/Rust/Python, Rust/Python/Triton. Every measured implementation in every
round passes a full-output comparison (`rtol=atol=1e-4`). Means use all 300
samples. Plotted whiskers are the minimum and maximum of the three round
means, not confidence intervals or request latency percentiles.

| Operation | PyTorch (µs) | Triton (µs) | Rust (µs) |
| --- | ---: | ---: | ---: |
| Matmul | 50.93 | 88.82 | 326.72 |
| Bias + GELU | 27.70 | 24.29 | 12.95 |
| LayerNorm | 17.77 | 22.61 | 19.47 |
| Triangle contraction | 55.56 | 91.91 | 78.55 |
| Neighbor aggregation | 89.66 | 21.80 | 13.18 |

Inputs are resident on the device. CUDA events surround each operation and
the end event is synchronized each iteration. Rust and Triton reuse output
storage; the eager PyTorch expressions allocate outputs/intermediates through
PyTorch's allocator. Compilation, input transfer, and native process startup
are excluded. Host submission and allocation can leave GPU-idle gaps inside an
event span. These timings do not isolate device instructions from the host
launch path, and they are not end-to-end service timings.

### Nsight Systems

Fifteen new captures cover 100 iterations per operation and implementation:
2,100 kernel launches in total, persisted through Mage. The plotted GPU time
is the sum of all captured kernel durations divided by requested iterations,
so a multi-kernel PyTorch operation is compared to the complete custom operation.
The launch chart uses the same denominator.

| Operation | PyTorch GPU µs/op | Triton GPU µs/op | Rust GPU µs/op | Launches: PyTorch / Triton / Rust |
| --- | ---: | ---: | ---: | --- |
| Matmul | 55.96 | 83.78 | 343.99 | 1 / 1 / 1 |
| Bias + GELU | 15.91 | 7.94 | 11.21 | 2 / 1 / 1 |
| LayerNorm | 11.44 | 8.19 | 18.64 | 1 / 1 / 1 |
| Triangle contraction | 28.46 | 93.87 | 81.16 | 3 / 1 / 1 |
| Neighbor aggregation | 78.34 | 7.92 | 10.28 | 4 / 1 / 1 |

The GPU-only view excludes gaps between kernels, synchronization, allocations,
and API work. It is a **separate instrumented run**, not a decomposition of the
event samples. Python/Triton capture targets enqueue a sequence and synchronize
at the end; the Rust target records and synchronizes events per iteration.
These launch rhythms can affect clocks, caches, and profiler overhead. Do not
subtract the two charts to estimate host overhead or compare their totals as
though they came from a single timeline.

This WSL host still uses NVIDIA's `CuptiUseRawGpuTimestamps=false` fallback,
which has reduced timestamp precision. GPU clocks were not locked; there was
no exclusive-use guarantee for this desktop GPU. This is one workstation
session with three event rounds and one capture per implementation/operation.
Nsight Compute hardware counters remain unavailable; no occupancy, cache-hit,
bandwidth, or bottleneck diagnosis is inferred from this data.

## Validation

- All 45 measured operation/implementation/round outputs passed full-element checks.
- The Triton correctness suite passed all 15 cases: five default shapes, five
  tile-tail shapes, constant/single-feature LayerNorm, empty adjacency,
  duplicate edges, and a single-element matmul.
- The same 15 tests passed under Compute Sanitizer memcheck: **0 errors**.
- Every Rust and Triton capture contains exactly 100 launches. Existing Python
  and Rust capture modes remain the default; Triton is an explicit additional target.

## Reproduce

After the existing [environment and native build setup](../guide.md), run from
the repository root. Choose fresh output paths; the comparison runner rejects
an existing event output directory to preserve prior evidence.

```bash
.venv/bin/python -m pytest tests/test_triton_comparison.py -q
.venv/bin/python examples/oxide/comparison.py --output artifacts/comparison-new
.venv/bin/python examples/oxide/profile_suite.py \
  --inputs artifacts/comparison-new --output artifacts/comparison-new-nsys \
  --languages python triton rust
.venv/bin/python examples/oxide/export_comparison.py \
  --events artifacts/comparison-new --profiles artifacts/comparison-new-nsys
uv run --script scripts/plot-comparison.py
```

The plotting script pins Matplotlib, validates the sample counts and correctness
flags, derives values from retained evidence, and generates desktop/mobile SVGs,
downloadable PNGs, and the accessible numeric tables. Pages only needs the
committed artifacts. Regenerating graphs for a new run also requires updating
the prose and interpretation; do not carry forward these conclusions blindly.

- [All event samples, checks, run order, input/source/binary hashes, GPU state](../assets/results/mage-001/comparison-results.json)
- [Capture inventory and all normalized kernel durations](../assets/results/mage-001/comparison-profiles.json)
- [Original two-implementation experiment](mage-001-validation.md)

## Production decision to test next

The current measurements support keeping the existing library-backed dense
operations and investigating Triton fusion inside a PyTorch application. Rust
may still make sense in a native application where integration and control of
the launch path are valuable. This experiment cannot establish that Rust or
Triton is generally faster, and does not establish production readiness.

Before replacing an operation, measure representative shapes and dtypes in the
real call path, including compiled/graph execution, batching, memory lifetimes,
transfers, throughput, p50/p95/p99 latency, and contention. Test backward
operations if training is required. Use deployment GPUs, a supported pinned
toolchain, correctness tolerances appropriate to the application, and a fallback.

PyTorch documents [Triton integration and autograd registration](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).
cuda-oxide currently identifies itself as an [experimental alpha compiler](https://github.com/NVlabs/cuda-oxide#project-status).
