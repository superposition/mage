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
7. accept only if the median of the **paired per-round ratios** clears `--min-gain`
   (default 1%) **and** the committed control's round medians stayed within
   `--control-drift` (default 3%). Pairing matters: clock-boost excursions of about 10%
   land on either arm, so a rule built on the fastest round of each arm can be carried by
   a single boosted round, while a paired median is perturbed in one ratio out of R;
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

The first confirmations of this run read +4.36% and +4.94%, and **both overstated the
gain**. The confirmation step now uses the protocol that measures it:

- two prepared input directories per arm, reused across rounds, so no round pays its own
  cold-cache cost;
- four unrecorded warm-up pairs, because the first measurement after an idle period reads
  several percent fast;
- eight recorded rounds with the arm order alternating, 150 iterations each;
- the quoted figure is the **median of the per-round ratios**, with the mean, range and
  the count of rounds favouring the incumbent recorded beside it
  (`--confirm-rounds`, `--warmup-pairs` in `scripts/evolve.py`).

Under that protocol the retained configuration reads **80.90 µs against 82.94 µs, median
+2.47%, 8 of 8 rounds in favour of the incumbent**, maximum absolute error 0.0. The same
pair read 0.9753 in seven of eight rounds; one round read 0.8846.

The mean of the same eight rounds is +4.12%, and that spread is the second finding: the
machine produces occasional clock-boost excursions of about 10% that land on **either**
arm. In that run the incumbent read 72.70 µs once and the committed kernel read 77.82 µs
once, against 80.90 and 82.94 µs otherwise. So a mean over a few rounds, or a rule that
takes the fastest round on each arm, is biased by whichever arm catches the boost first.
The median over eight alternating rounds is stable to 0.4% across sessions; the mean is
not. The accepted steps are unaffected in direction: their ratios (0.778, 0.912, 0.924)
are an order of magnitude outside the excursions, and the same three steps were accepted
with a different seed. Only the quoted total moved, from about 5% to about 2.5%.

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

## Kernel time: does the event-span win survive the published instrument?

The mage-001..003 records are GPU kernel time from Nsight Systems captures, so the
retained configuration was re-measured with that instrument before any comparison with
them. `scripts/evolve_capture.py` reuses the harness path of
`examples/oxide/profile_suite.py` (`NsysBackend`, `capture_range="cuda"`, the binary's own
`--capture` bracket), captures 100 iterations per arm, interleaves the arms, and asserts
that both arms received byte-identical inputs.

**The committed kernel reproduced the published record**: 80.2-80.6 µs per iteration
against mage-003's 80.00 µs. The retained configuration measured **76.29 µs, 0.951 of the
committed kernel — a 4.9% improvement in GPU kernel time**, not only in the event span.

It is an interaction, not a single knob. Each row below is one paired capture session at
1024³ FP32, kernel microseconds per iteration, arms interleaved:

| staging structure | `k_step` | kernel µs | vs committed |
| --- | ---: | ---: | ---: |
| `shared` (the committed kernel) | 64 | 80.31 | 1.000 |
| `exact` (guard-free, one loop) | 64 | 80.29 | 0.999 |
| `exact` | 32 | 82.43 | 1.028 |
| `guarded` (one branch per tile) | 64 | 82.97 | 1.034 |
| `guarded` | 32 | **76.29** | **0.951** |

`k_step` 32 is *slower* in the guard-free structure (+2.6%, the direction the published
32 -> 64 step recorded) and *faster* in the guarded structure (-8.0%); guarded staging at
`k_step` 64 is itself slower than the committed kernel. Neither knob alone explains the
result and the loop could not have reached it by measuring either alone — the accepted
steps were three single-knob changes in sequence, and only the combination pays.

One session disagreed. The first capture ran about sixty times slower than every later
one (846 s against 14 s for the same work) and read the retained arm at 80.4 µs with the
committed arm normal at 79.2-80.3 µs, which would have meant no kernel-time gain at all.
Four later sessions, each with its own build, agreed with each other at 74.8-76.5 µs for
the retained arm, and the structural comparisons above reproduce inside single sessions.
The anomalous session is recorded here rather than dropped:
`docs/assets/results/evolution-loop/kernel-time/`.

Against the published baselines: cuBLAS at 44.03 µs is still ahead, and Triton's matmul
kernel swings 71.94-83.78 µs between sessions, so no ranking follows from a single
pairing. A same-session Triton capture is the outstanding measurement.

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
- The retained configuration is measured in both instruments now: 0.951 of the committed
  kernel in GPU kernel time (Nsight, three paired captures) and a median +2.47% in event
  span over eight paired rounds. It is still one shape (1024³), one dtype (FP32), one GPU,
  and it has no Triton pairing inside a session.
- No hardware counters were available, so no bounded-resource explanation accompanies the
  accepted steps; the loop reports what changed and by how much, not why.
- Coordinate descent proposes one knob at a time. Coupled moves (a larger tile needs a
  smaller k step to stay inside the shared-memory budget) are only reachable through the
  staging repair, which pairs an invalid geometry proposal with the mode that fits it.
- The retained configuration is a candidate, not a published result: it has not been
  captured with Nsight, has not been confirmed in a fresh session, and does not carry the
  correctness enumeration (fallback kernels, edge shapes) that the mage-003 record does.
- The per-generation accept rule still compares the fastest observed round on each arm.
  That is deliberate — it is a screen for reversals, and it refuses to judge a generation
  whose control drifts more than 3% — but it inherits the clock-boost bias: one boosted
  round on the candidate arm can carry a generation. With the ratios this run accepted
  (0.778, 0.912, 0.924) the screen is not close to that bias; a rule that admits smaller
  gains should compare per-round ratios instead.
- The confirmation measures the *event span around the call*, on one shape, in one session.
  Its median ratio is stable to 0.4%, but the excursions described above mean a single
  round is not a measurement.

## Next

- Make the confirmation step use warm-up pairs and more rounds, so a run cannot quote a
  gain its own protocol inflated.
- Capture Triton in the same session as the retained kernel. The published Triton matmul
  value swings between 71.94 and 83.78 µs across sessions, so a cross-session comparison
  says nothing; a same-session pairing is the only way to rank them.
- Feed the loop the profiler's own output (Mage already captures Systems and Compute
  reports) so proposals can be diagnosis-driven rather than order-driven.
- Let the proposer write structure, not just constants: the generated source is already
  the only artifact the loop edits, so a code-writing proposer fits behind the same gate
  and ledger.
