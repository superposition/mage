# mage-007: what three kernel changes measured

Status: **two improvements and one rejection**, measured 10 September 2026. This
record belongs to the cuda-oxide kernel lane, which the cuTile track picked up when
its agent exited; mage-006 stays as that agent left it.

The lane's ranked gaps were wide-row LayerNorm, then GELU and neighbor aggregation.
This round took all three: one closed, one halved, one rejected with its numbers
kept.

## Method

| | |
| --- | --- |
| Instrument | GPU kernel time, one Nsight Systems capture per measurement, mean of 100 launches after 25 warm-up launches |
| Host | RTX 4090 (sm_89), driver 591.74, WSL2 Ubuntu 22.04, device idle |
| Session discipline | Each before/after pair was measured in one session, with the same inputs and the same warm-up, on the same binary path; the device was checked idle before each pass |
| Correctness | Full-output check against PyTorch FP32 with TF32 disabled, `rtol = atol = 1e-4`, before any timing was believed |
| References | Triton and cuTile bars are quoted from their own sessions (mage-004, mage-006) and are drawn hatched in the figure because they are not pairings |

The figure is [`docs/assets/figures/mage-007/kernel-lane-fixes.svg`](../assets/figures/mage-007/kernel-lane-fixes.svg),
drawn by `scripts/plot-kernel-lane-fixes.py` from
[`docs/assets/results/mage-007/kernel-lane-fixes.json`](../assets/results/mage-007/kernel-lane-fixes.json).

## Values

Kernel time in microseconds, mean of 100 launches.

| Operation | Before | After | Change | Reference (other session) |
| --- | ---: | ---: | ---: | --- |
| LayerNorm 4096×4096 | 255.41 (`layer_norm_warp`) | **186.80** (`layer_norm_pair`) | 1.37× faster | Triton 181.59 (mage-006) |
| Neighbor aggregation 4096×64×65536 | 9.45 (scalar) | **7.06** (feature quad) | 1.34× faster | Triton 5.97, PyTorch 120.93 (mage-006) |
| Bias + GELU 4096×768 | 10.29 (scalar, kept) | 27.00 (feature quad, **rejected**) | 2.6× slower | Triton 7.76, cuTile 7.97 (mage-004) |

Worst full-output error: 4.77e-07 (LayerNorm), 3.58e-07 (neighbor), 5.31e-06 (GELU).

## What each change was

**LayerNorm at wide rows** (issue #54, PR #69). The two-warp row kernel was gated at
`width <= 2048`, so a 4096-wide row fell back to `layer_norm_warp` — the capture named
the kernel, which is how the fallback was confirmed rather than assumed. The pairwise
kernel's own soundness condition, each warp's span a multiple of 32 lanes, already
held at 4096: the cap was a stale heuristic. Raising it to 4096 is the whole change.

**Neighbor aggregation** (issue #59, PR #71). The scalar kernel gives each thread one
feature and a serial edge loop, so one gather is in flight per thread. The quad form
gives each thread four features — one 128-bit load per edge — and unrolls the edge
walk by two, so two independent gathers overlap. Triton's 5.97 µs remains ahead, so
the gap narrows from 1.7× to 1.2×; the kernel moves about 17 MB of gathered rows in
7 µs, which is L2-bandwidth work on this part, so more edges in flight per thread is
the next lever.

**Bias + GELU** (issue #59, PR #71). The same vectorization was tried and rejected:
four features per thread measured 27.00 µs against the scalar kernel's 10.29 µs in the
same session. The LayerNorm treatment does not transfer to this elementwise shape —
one element per thread wins, most likely on occupancy. Kept here because it is the
counterexample: "vectorize the elementwise kernel" is not a rule.

## What the numbers do not establish

- No hardware counters are available, so the L2-bandwidth reading of the neighbor
  kernel is inferred from traffic arithmetic, not measured.
- One capture per point: a kernel time here carries no interval of its own.
- The references are quoted from other sessions. A consolidated pass
  (`scripts/evolve_capture.py --all`) is built for same-session comparisons and would
  make these four arms one measurement whenever the lane wants it.
- The LayerNorm change is validated at 4096×4096, 4096×512, 3072×1024 and 2048×2048;
  widths above 4096 still fall back, and were not measured.
- GELU's rejection is a single before/after pair, not a sweep: other vector widths
  (two features, or quads with a different block size) were not tried.

## Reproduction

```bash
source scripts/oxide-env.sh
cd examples/oxide && CARGO_BUILD_JOBS=4 cargo oxide build --arch sm_89 && cd ../..
# one operation, one capture
.venv/bin/python -m mage profile-exec ... # see docs/profiling.md for the native workflow
uv run --script scripts/plot-kernel-lane-fixes.py
```
