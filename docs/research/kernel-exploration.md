# Continuing the kernel exploration

Status: **the rewritten Rust kernels are measured and published as mage-002; the launch-path comparison below is still to come**.

[Journal](https://superposition.github.io/) ·
[What the first profiles taught us](https://superposition.github.io/journal/faster-calls-slower-kernels/) ·
[Rewriting the layer norm and matmul kernels](https://superposition.github.io/journal/rewriting-the-layer-norm-and-matmul-kernels/) ·
[Current graphs](https://superposition.github.io/mage/experiments/mage-001/#profiles) ·
[Evidence and method](../experiments/mage-001-comparison.md)

## The question

Which changes to the structure and execution of a kernel improve a real
workload, and can we explain why? Keep Python/Triton and native Rust profiling
available throughout this work. Preserve the first versions so improvements
and regressions remain inspectable.

The first cuda-oxide and fixed-tile Triton kernels established correctness and
a working profiling loop. They did not establish a language winner. In the
captured GPU durations, Triton led GELU, LayerNorm, and neighbor aggregation;
PyTorch's library-backed operations led matmul and triangle contraction. Rust
had shorter event spans for GELU and neighbor aggregation despite slower GPU
kernels. The event samples and Systems captures have different execution
rhythms, so their difference does not isolate host overhead.

The existing results cover five fixed FP32 shapes on one WSL RTX 4090 with
unlocked clocks. They exclude compilation, transfers, and service startup.
Training backward passes, tuned variants, compiled PyTorch, graph replay, and
end-to-end production behavior have not been compared.

## Start with mage-002: separate the kernel from its launch path

Use fused bias + GELU as the first controlled case. It is simple enough to
explain, has a shared mathematical contract, and exposes the reversal between
event spans and GPU durations.

1. Preserve the current benchmark as `mage-001`. Create an experiment directory
   and result namespace for `mage-002`; do not overwrite the published inputs,
   samples, graphs, or first kernel implementations.
2. Keep eager PyTorch as a reference and add a warmed `torch.compile` baseline.
   Record compilation separately. Verify the pinned versions' actual support
   before adding CUDA graph replay, including the Rust runtime's capture and
   launch APIs. An unavailable mode should be reported as unavailable.
3. Use explicit timing modes: single-operation event span; a batch of launches
   timed together and divided by repetitions; and graph replay where supported.
   Record how many operations a graph contains and distinguish replay latency
   from time per operation. Use the same stream, synchronization placement,
   input residence, and output allocation policy when comparing modes; document
   any remaining differences. Measure end-to-end host latency separately.
4. Sweep small, medium, and large inputs, including tile tails. Begin with a
   compact bounded set rather than a large search. Keep the FP32 contract fixed
   and test every output before collecting accepted timings. Warm JIT and
   autotuning outside the timed region, and rotate measurement order across
   independent rounds. Retain samples and variation, not only a winning mean.
5. Capture representative cases with Nsight Systems. Inspect the actual launch
   sequence and allocation/copy behavior. Use Compute Sanitizer on changed
   kernels. Keep hardware-counter fields unknown while access is unavailable.

**Completion:** a rerunnable comparison explains whether the apparent native
advantage persists after matching the launch path. A null result or a reversal
is a valid outcome. Publish the interpretation with graphs, then link the full
method and unsuccessful variants from GitHub.

## Then change one part of the computation

| Investigation | Hypothesis to test | What must stay comparable |
| --- | --- | --- |
| LayerNorm reduction | Warp/block reduction layout, row grouping, or vectorized loads can reduce synchronization and memory work. | Centered variance, epsilon, affine transform, dtype, shapes, full-output checks. |
| Neighbor aggregation | Scheduling around degree distribution changes load balance; a uniform graph can hide costly skew. | Identical CSR bytes, weights, output definition, empty rows and duplicate edges. Sweep degree distributions as well as sizes. |
| Matmul and triangle contraction | Layout, tiling, and reuse may matter more than the source language. | First use IEEE FP32 and the same contraction. Preserve library baselines and include layout conversion costs. |
| Lower-precision dense kernels | Tensor-core-friendly representations may change the useful algorithm and error budget. | Treat FP16/BF16/TF32 as separate contracts with explicit tolerances and error measurements, never as an undisclosed replacement for strict FP32. Verify sm_89 support. |
| Fused application fragment | Eliminating an intermediate or launch may help more than tuning an isolated primitive. | Compare the complete equivalent computation, memory footprint, precision, and realistic call path. |

Do not tackle all five directions at once. Use the controlled GELU comparison
to choose the next bottleneck, state the hypothesis before tuning, and retain
the first version beside each candidate. Avoid selecting a configuration on
the same samples used to report its performance; confirm it in fresh runs.

## Mage work that supports the research

- Generalize the experiment runner around explicit implementation/variant IDs,
  shapes, dtype and precision mode, input hashes, source/binary hashes, timing
  mode, repetitions, warmup, round/order, device state, and correctness. Preserve
  the existing CLI and saved history while adding fields or result formats.
- Keep event spans, per-kernel durations, complete-operation sums, and host
  latency as different measurements. Never average a multi-kernel operation's
  launches and label that number as its complete cost.
- Generate comparisons from retained samples. Label the baseline, shape,
  units, uncertainty, and measurement boundary; show regressions. Keep numeric
  tables accessible and graph exports readable without local fonts.
- Keep failure/empty-capture handling, exact argv, unique report directories,
  Python interpreter selection, and profiler-hook restoration intact. Extend
  the relevant regression checks when a change touches those behaviors.

Start from `examples/oxide/comparison.py`, `triton_target.py`, `profile_suite.py`,
and `export_comparison.py`; the native kernels are in `examples/oxide/src/main.rs`.
The shared contracts are in [kernel-contracts.md](../kernel-contracts.md).
`scripts/plot-comparison.py` and `scripts/plot-kernel-improvements.py` take their
experiment namespace as `--experiment` (default mage-001 and mage-002), so later
measurements can be plotted without editing the scripts.

## Before calling it a production improvement

Choose a concrete model or service with the user. Measure its representative
shapes, precision, batching, memory behavior, transfers, throughput, and latency
distribution on the deployment GPU. Include backward correctness and performance
if training matters. A faster kernel is useful only to the extent that the
workload spends time in it. Keep a reference implementation and a fallback.

The public journal should explain the question, the mathematical or scientific
connection, the observation, and the next question. Keep command transcripts,
configuration matrices, test logs, and exact contracts in GitHub. Maintain the
compact dark layout and the links between the journal, Mage field notes, README,
and reproducible evidence. Every stage should leave both code and a clearer
account of what was learned.
