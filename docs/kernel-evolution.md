# Kernel evolution: a closed measurement loop over generated variants

Status: **implemented and exercised on matrix multiplication.** The loop generates a
kernel variant, builds it, gates it on the PyTorch reference, measures it against both
the incumbent and the committed kernel, and records the decision. Three of twelve
generations were accepted in the first run; the retained configuration measured 78.9 µs
of event span against the committed kernel's 82.9 µs on the same inputs and shape.

## What "self-improving" means here

The phrase is used at three levels, and only the first is implemented:

1. **Search.** A measured result chooses the next variant. Nothing about the harness
   changes; the loop closes measurement onto proposal.
2. **Diagnosis.** The *kind* of change is chosen from an explanation of the last result
   (which resource the kernel is short of), not from a fixed parameter order. The
   mage-003 record was produced this way by hand: a deliberately worse tile separated
   shared-memory instructions from arithmetic, and the resident-thread count chose the
   layer-normalization change.
3. **Recursion.** The proposal policy itself is the object of improvement, using the
   accumulated ledger (which kind of change paid, which is exhausted) and, at the far
   end, generating new kernel structure rather than new constants.

A benchmark cannot verify its way out of a bad verifier, so the loop's verifier is
deliberately the fixed part: the committed kernels and the PyTorch FP32 reference are
never generated, never skipped, and never re-tuned by the loop.

## The loop

Per generation, one build and three paired measurements per round:

1. propose one knob change against the incumbent (`scripts/evolve_policy.py`);
2. render both roles from the same templates (`scripts/evolve_render.py`);
3. build (`cargo oxide build --arch sm_89`, 1.9 s incremental);
4. reject statically if the validator already knows the geometry cannot work
   (shared-memory budget, alignment, thread count, staging precondition) — this costs no
   build;
5. measure `best`, `new` and the committed kernel in the same round, rotating the order
   between rounds; inputs are generated once, hashed, and reused;
6. gate on the full-output comparison against the PyTorch FP32 reference before any
   timing is read;
7. accept only if the candidate's worst round median beats the incumbent's best round
   median by more than `--min-gain` (default 1%) **and** the committed control's round
   medians stayed within `--control-drift` (default 3%);
8. append one ledger entry with the parameters, per-round numbers, correctness, decision
   and reason.

`--rounds 1` makes the drift guard vacuous (a single control sample reads 1.0x); the
first run used three rounds and observed 1.001-1.013 drift, so the guard was doing work.

## The variant surface

`examples/oxide/src/candidates.rs` is generated from `candidates.rs.tmpl` and
`candidates_body.rs.tmpl`. Two roles are rendered per build, `best` (incumbent) and `new`
(proposal), so one build measures both under identical conditions. The manifest selects
one:

```json
{"op": "matmul", "dims": [1024, 1024, 1024], "warmup": 10, "iterations": 60, "variant": "best"}
```

A manifest with no `variant` runs the committed kernels in `src/main.rs`; those are
never modified by the loop, so the published records stay reproducible. Every timing
record now names the kernel that produced it (`variant`, `variant_kernel`,
`variant_params`), including which committed kernel the shape selected.

## Search space and static constraints

| Operation | Knobs | Values |
| --- | --- | --- |
| matmul | `transpose_a`, `quad_stage` | boolean staging layout choices |
| matmul | `staging` | `shared` (one quad decomposition, needs block_m = block_n = k_step), `exact` (guard-free, both tiles fill the same whole number of passes), `guarded` (fits every geometry, one branch per tile) |
| matmul | `k_step` | 16, 32, 64, 128 |
| matmul | `block`, `thread_tile` | tile shape and per-thread register tile |
| layernorm | `warps_per_row`, `threads` | 1, 2, 4, 8 warps per row; 128, 256, 512 threads |

Rejected before any build: more than 46 080 B of static shared memory, thread counts
that are not whole warps, 128-bit access that is not 4-element aligned, and any
guard-free form whose quad counts do not divide. `k_step=128` at the default tile is
rejected this way (67 584 B).

## First run

Command (from the repository root, Python environment with PyTorch active):

```bash
.venv/bin/python scripts/evolve.py --op matmul --generations 12 --rounds 3 \
  --iterations 60 --max-stale 8 \
  --start-json '{"k_step": 64, "transpose_a": false, "quad_stage": false, "staging": "guarded"}' \
  --out artifacts/evolution/matmul-loop-demo
```

The start configuration is the pre-optimization arrangement of the mage-001 kernel family
(row-major A staging, scalar staging loads) at 1024³. Worst-of-round medians, event span
around the call:

| Gen | Change | Incumbent µs | Candidate µs | Ratio | Drift | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `transpose_a` false -> true | 119.81 | 93.18 | 0.778 | 1.010 | **accept** |
| 2 | `quad_stage` false -> true | 93.18 | 84.99 | 0.912 | 1.001 | **accept** |
| 4 | `k_step` 64 -> 32 | 84.99 | 78.50 | 0.924 | 1.003 | **accept** |
| 3 | `transpose_a` true -> false | 84.99 | 107.52 | 1.274 | 1.004 | reject |
| 5 | `transpose_a` true -> false | 78.66 | 81.92 | 1.042 | 1.010 | reject |
| 6 | `quad_stage` true -> false | 78.58 | 87.04 | 1.107 | 1.013 | reject |
| 7 | `k_step` 32 -> 16 | 78.59 | 82.94 | 1.055 | 1.004 | reject |
| 8 | `k_step` 32 -> 128 | - | - | - | - | rejected statically (shared budget) |
| 9 | `block` 64x64 -> 128x64 | 78.48 | 83.81 | 1.066 | 1.009 | reject |
| 10 | `block` 64x64 -> 64x128 | 77.84 | 84.35 | 1.077 | 1.009 | reject |
| 11 | `staging` guarded -> shared | 78.50 | 84.02 | 1.068 | 1.006 | reject |
| 12 | `staging` guarded -> exact | 78.67 | 84.61 | 1.074 | 1.009 | reject |

The run stopped after eight consecutive rejections. Final parameters:
`{block: 64x64, k_step: 32, transpose_a: true, quad_stage: true, staging: guarded,
thread_tile: 4x4}`.

The confirmation step re-measured the final incumbent against the committed kernel in
fresh processes: **78.70 µs against 82.29 µs (+4.36%)**. An independent re-measurement in
three further alternating rounds, with the final parameters rendered into the generated
slot and the committed kernels reached through the no-variant path, read **78.85 µs
against 82.93 µs (+4.94%)**, maximum absolute error 0.0, sample-to-sample spread within
0.1 µs. The baseline itself reads 81.9-82.9 µs across these sessions, which bounds how
much of the difference any single pairing can attribute.

Ledger and summary for the matrix-multiply run are committed under
`docs/assets/results/evolution-loop/matmul-ledger.jsonl` and `matmul-summary.json`.

## Correctness findings

The gate rejected candidates for three distinct reasons during integration, all now
fixed in the generator:

- **Per-lane shuffle without broadcast.** The single-warp form used the value left in
  each lane after the shuffle-down reduction. Only lane 0 holds the warp total, so every
  lane normalized with a different denominator (maximum absolute error 1.03). The
  committed single-warp kernel broadcasts with `shuffle_f32(sum, 0)`; the template now
  does the same.
- **Overlapping warp shares.** Splitting a row across warps by whole 32-lane steps
  double-counts when the per-warp span is not a multiple of 32. Width 768 (span 96) is
  safe, which is why the published captures never saw it; width 128 (span 16) is not. The
  generated template now restricts each warp to its own `[start, end)` range, which is
  correct at every width and warps-per-row (checked at widths 32, 128, 768, 2048 with 1,
  2, 4 and 8 warps per row).

**The committed `layer_norm_pair` shared this defect** for widths below 256 that are
multiples of four, which the published captures never exercised. It is fixed in the host
dispatch: `layer_norm_pair` is now selected only when each warp's span is a whole number
of 32-lane steps, and narrower rows take the single-warp kernel, which guards every
access. Verified after the change: widths 4, 64 and 128 report `layer_norm_warp` and pass
the reference; widths 252, 256, 768 and 2048 report `layer_norm_pair` and pass; width 257
takes the scalar path and passes; and the full small suite (five operations, every edge
shape, including the `[17, 19, 23]` matmul fallback and `[3, 257]` layer norm) passes. The
published records are unaffected, since width 768 and the scalar 257 path route exactly as
before.
- **Undersized shared array.** The transposed A layout needs `(block_m + 4) x k_step`
  floats and the row-major layout needs `block_m x k_step`; the generator declared
  `at_stride x k_step` for both. At 64x32 that halved the array, so A writes overwrote
  the adjacent B tile (maximum absolute error 2.84 at 256x256x128, and an illegal memory
  access at 128x64).

The search space is now verified as a set: 40 combinations of block, staging layout,
staging mode and k_step are all correct at 256x256x128, and the layernorm family is
correct across its whole knob set.

## LayerNorm: a null result

The same loop on layer normalization (`--op layernorm`, starting from the committed
configuration) ran four generations and accepted none. Worst-of-round medians, event span
around the call, against the incumbent:

| Gen | Change | Incumbent µs | Candidate µs | Ratio | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `warps_per_row` 2 -> 1 | 12.29 | 13.07 | 1.064 | reject (control drifted 12.5%, so the round is not judged) |
| 2 | `warps_per_row` 2 -> 4 | 12.29 | 12.29 | 1.000 | reject |
| 3 | `threads` 256 -> 128 | 12.29 | 12.29 | 1.000 | reject (0.52% gain, below the 1% threshold) |
| 4 | `threads` 256 -> 512 | 12.29 | 13.31 | 1.083 | reject |

The committed configuration (two warps per row, 256 threads) is a local optimum in this
two-knob space, and the loop reports that instead of manufacturing an improvement. Note
the first generation: at a 12 µs span the committed control is sensitive enough that a
12.5% excursion occurred inside one generation, and the loop refused to judge those
rounds rather than accept a candidate whose win would have been clock noise.

The remaining distance to Triton's layer-normalization kernel (10.05 µs against 7.99 µs of
GPU kernel time in mage-003) is not reachable by re-assigning the existing work; it needs a
different decomposition, which is the case for a proposer that can write structure rather
than select constants.

Ledger and summary: `docs/assets/results/evolution-loop/layernorm-ledger.jsonl` and
`layernorm-summary.json`.

## What the loop does not establish

- One shape (1024³), one dtype (FP32), one GPU (RTX 4090, unlocked clocks, WSL).
- The retained number is a **CUDA event span around the call**, which includes host
  submission; it is not the GPU kernel duration that the mage-001..003 records use. The
  `k_step 32` result in particular needs its own Nsight capture before it can be compared
  with the published 80.0 µs kernel time.
- No hardware counters were available, so no bounded-resource explanation accompanies the
  accepted steps; the loop reports what changed and by how much, not why.
- Coordinate descent proposes one knob at a time. Coupled moves (a larger tile needs a
  smaller k step to stay inside the shared-memory budget) are only reachable through the
  staging repair, which pairs an invalid geometry proposal with the mode that fits it.
- The retained configuration is a candidate, not a published result: it has not been
  captured with Nsight, has not been confirmed in a fresh session, and does not carry the
  correctness enumeration (fallback kernels, edge shapes) that the mage-003 record does.

## Next

- Nsight-capture the retained configuration so the claim can be stated in kernel time.
- Feed the loop the profiler's own output (Mage already captures Systems and Compute
  reports) so proposals can be diagnosis-driven rather than order-driven.
- Let the proposer write structure, not just constants: the generated source is already
  the only artifact the loop edits, so a code-writing proposer fits behind the same gate
  and ledger.
