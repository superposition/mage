# Parallel tracks — handoff

Several agents work this repository at once. This file is the shared map: who owns what, which resources are contended, what the current numbers are, and what each track should do next. Read it before starting work; each track also keeps its own brief (`docs/research/kernel-exploration.md` for the cuda-oxide kernel work, `docs/research/cutile-rust.md` for the cuTile work).

Coordination index and protocols: mage issue #53. Open performance gaps: issues #54 and #59.

## Tracks

| Track | Lanes it owns | Worktree and branch | Published record |
| --- | --- | --- | --- |
| cuda-oxide kernel optimisation | `examples/oxide/src/main.rs` kernels, `examples/oxide/*` harness targets, `scripts/plot-comparison.py` | `/home/superposition/code/mage-perf` (detached or its own branch) | mage-006 (#58) |
| cuTile Rust | `examples/cutile/**`, `docs/research/cutile-rust.md` | `/home/superposition/code/mage-cutile` (`feat/cutile-rust`, #55 merged) | mage-004 (#55) |
| Closed measurement loop | `scripts/evolve*.py`, `examples/oxide/src/candidates.rs`, `tests/test_evolution.py`, and the `committed_kernel` hooks in `examples/oxide/src/main.rs` | `/home/superposition/code/mage-oxide` (all merged: #48, #51, #57) | mage-005 (#50 merged; journal entry and field note are live) |
| Publication | `docs/experiments/**`, `docs/assets/{results,figures}/**`, blog posts | one worktree per stage record | mage-002, 003, 006 (#45, #46/#47, #58) |

`docs/experiments/index.md` is an append-only list shared by every record. Expect a conflict there whenever two records land close together; the resolution is to keep both entries in order.

## Shared resources and the rules that keep them usable

**One GPU, one measurement at a time.** Before and after every run:

```
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader
```

Sustained utilisation above ~30% or power above ~100 W with nothing of your own running means another track is measuring, and any timing taken then is noise. Include a control in every session — the Triton or PyTorch kernel whose value you already know — and discard the run if the control lands outside its band rather than publishing it with a caveat. That has already caught one contaminated capture (Triton matmul at 100.80 µs against its 71.94-90.09 µs band) and one wrong "win" (a cp.async pipeline that measured 69 µs while computing the wrong answer).

**One worktree per track.** Build only in your own worktree; `target/release/mage-oxide` is a shared path inside a worktree, and rebuilding over another track's binary invalidates their run.

**`examples/oxide/src/main.rs` is shared** between the kernel track and the loop track. The dispatch resolves `committed_kernel` once per shape and then matches on it, so a new kernel is added as a new value in that resolution plus a new arm in the launch match. Keep the resolution single-sourced; when a conflict lands, keep both arms.

**Claim a namespace, then re-check it.** A stage record claims a number, `docs/assets/results/mage-00N/`, `docs/assets/figures/mage-00N/`, a field note and a blog `experiment_id`. Claim it in issue #53, then re-check immediately before publishing, because a claim written earlier can be taken by a PR that lands later — that is how mage-004 went to the cuTile track and the kernel track's record became mage-006:

```
gh api repos/superposition/mage/contents/docs/experiments --jq '.[].name'
```

**Evidence layout.** Raw captures stay local and gitignored under `artifacts/`; the portable evidence is committed under `docs/assets/results/<namespace>/` and must not contain absolute home paths (`export_comparison.py` rewrites capture directories relative to the checkout). Figures follow the existing palette and outline their glyphs, so every figure needs an accessible value table beside it.

**Blog.** One branch per entry, never another track's branch. Entries state which metric each number is: kernel time from Nsight Systems, or the span around the call from CUDA events. The cuTile record showed why that matters — a lazy runtime can price its host submission path at ~16 µs per launch against 2-3 µs for the hand-written one, so the span column can be dominated by the launch path rather than the kernel.

**Merges.** Never merge by assumed PR number:

```
gh pr view <n> --repo superposition/mage --json number,title,headRefName,headRefOid
gh pr merge <n> --repo superposition/mage --squash --match-head-commit <headRefOid>
```

## The closed measurement loop

The loop renders a kernel from parameters into `examples/oxide/src/candidates.rs` — two roles,
`best` and `new`, in one build — builds it, checks the full output against the PyTorch
reference, measures it against both the incumbent and the committed kernel in the same round
with the order rotating, decides, and appends a ledger entry. It never edits the committed
kernels: a manifest `variant` field selects the generated role and a manifest without one
runs the committed kernels, so every earlier record stays reproducible.

```
# search around a configuration; the pipelined form is a staging value
.venv/bin/python scripts/evolve.py --op matmul --generations 12 --rounds 3 --iterations 60 \
  --start-json '{"staging": "pipeline", "k_step": 32, "block": [64, 64], "transpose_a": true, "quad_stage": true}' \
  --out artifacts/evolution/<name>

# after touching a template or the renderer: prove every reachable configuration is correct
.venv/bin/python scripts/evolve_audit.py --op matmul

# kernel time for one configuration, with the other implementations in the same session
.venv/bin/python scripts/evolve_capture.py --op matmul --triton --pytorch \
  --params '{"staging": "pipeline", "k_step": 32, "quad_stage": true, "block": [64, 64], "transpose_a": true, "thread_tile": [4, 4]}'
```

What it knows: the first run's retained configuration (guarded staging at `k_step` 32) was an
**interaction** — `k_step` 32 is slower in the guard-free structure and faster in the guarded
one — and it is superseded by the pipelined kernel, which the generated space now reproduces
at 0.989 of it in kernel time. Searching around the pipeline found a **flat neighbourhood**:
the width of the `A` copies and `quad_stage` move it by 0.1-0.3% where the same knobs were
worth 1.6x before the copies overlapped the arithmetic, `k_step` 32 is confirmed, larger tiles
are worse, and the synchronous forms are 20% behind.

Next for this track: the **buffer count** (three buffers instead of two) is the one pipelined
axis not yet in the space. Everything else that is left needs structure the templates cannot
express — split-K for the matmul gap to the library, and the block-per-row register-held shape
(PR #52) for wide-row LayerNorm.

## Current numbers

Kernel time in microseconds, one Nsight Systems capture per operation and implementation, 100 iterations. Sources: mage-006 for the cuda-oxide column, mage-004 for cuTile, both with Triton and PyTorch measured in the same session.

| Operation | cuda-oxide | Triton | PyTorch | cuTile |
| --- | ---: | ---: | ---: | ---: |
| Matrix multiplication 1024³ | **68.62** | 79.84 | 66.22 | 131.56 |
| Bias + GELU 4096x768 | 11.02 | 7.76 | 16.12 | 8.19 |
| LayerNorm 4096x768 | 8.97 | 8.07 | 11.25 | 10.76 |
| Triangle contraction 128x32 | 80.83 | 100.45 | 28.57 | 116.08 |
| Neighbor aggregation | 10.28 | 5.97 | 120.93 | 35.37 |

The cuTile column is mage-004's refreshed session (`#61`, the tile tuned to
32x128x32); its earlier values in this table were 172.52 / 7.97 / 10.46 / 165.22 /
62.12, and the last two of those were *event spans*, not kernel times. The cuTile
column and the cuda-oxide column come from different sessions, so they should not
be differenced against each other without re-measuring both in one run.

Event spans (mean of 300 warmed samples, three rotating rounds, mage-006 namespace): matmul 74.09, GELU 13.96, LayerNorm 12.26, triangle 86.85, neighbor 15.88 for the cuda-oxide kernels, against Triton's 93.29 / 28.35 / 27.01 / 100.08 / 27.62.

## Open gaps, ranked by what they are worth

1. **LayerNorm at wide rows is the largest measured gap** (issue #54). At 4096x4096 the cuda-oxide kernel takes 255.45 µs against Triton's 181.59 (1.41x) because the register-held row kernel covers rows up to 1024 elements and that shape falls back to the older single-warp kernel. Extending the held structure to wider rows, or giving the wide-row path the same treatment, is the first thing to try.
2. **LayerNorm at the harness shape is shape-specific, not structural** (issue #54). 4096x768 is 7-10% behind Triton, but at 8192x768 with the same width the kernel is 26% *ahead*. The wide-row fix above may or may not move this; the remaining hypotheses are in the issue.
3. **Bias + GELU and neighbor aggregation were never optimised** (issue #59). They are the mage-001 kernels: 11.02 against Triton's 7.76 and cuTile's 7.97, and 10.28 against Triton's 5.97. Both are elementwise or reduction shaped, so the treatment that took LayerNorm from 18.64 to 8.97 should apply.
4. **Matmul leads Triton** but not the library: 68.62 against cuBLAS 66.22 in the same session, and 44.03-66.22 across sessions. Split-K is the untried lever; re-arranging the pipeline is not — the loop searched that neighbourhood and found it flat (0.1-0.3% across the knobs that used to matter, `k_step` 32 confirmed).

## Environment

| | |
| --- | --- |
| Host | RTX 4090 (sm_89), driver 591.74, WSL2 Ubuntu-22.04 |
| Toolchain | CUDA 13.0.3 (`source scripts/oxide-env.sh`), LLVM/Clang 21, Rust nightly-2026-08-28, cuda-oxide rev `26754ae52c26c097dc1c465a1e42c4c5d05a3d40` |
| Build | `source scripts/oxide-env.sh && cd examples/oxide && CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89` |
| Python | `/home/superposition/code/mage-oxide/.venv` (torch 2.9.1+cu128, triton 3.8.0) |
| Harness | `examples/oxide/experiment.py` (correctness), `comparison.py` (event spans, three rotating rounds), `profile_suite.py` (Nsight Systems captures), `export_comparison.py` (portable evidence) |
| Mage CLI | `/home/superposition/.local/bin/mage`; `profile-exec ... -- <binary> <args>` for native, `profile <script> -- <args>` for scripts |
| Profilers | Nsight Systems 2025.3.2 with `CuptiUseRawGpuTimestamps=false`; Nsight Compute counters blocked by `ERR_NVGPUCTRPERM` |
| Figures | `uv run --script scripts/plot-comparison.py --experiment <namespace>`; `scripts/plot-kernel-progression.py` carries the per-kernel stage history |

## Rules written down after they were learned the hard way

1. A merge by assumed PR number merged another track's PR (#48) while two of its commits were left behind. Read the head, then merge with the guard.
2. A namespace claimed in the tracker was taken by a PR that landed after the claim. Re-check master immediately before publishing.
3. A faster timing can come from a wrong kernel: the first cp.async pipeline measured 69 µs while producing wrong output, from a buffer-offset unit error and a final-tile buffer index of `tiles % 2` instead of `(tiles - 1) % 2`. Validate correctness before believing a number.
4. Measuring while another track measures silently inflates both sets. Control values are the only way to notice.
5. A generated form has to be audited before it is searched. The space audit caught a shared array sized for the wrong A layout (wrong values at 256x256x128, an illegal access at 128x64) and a row-major branch that multiplied an array by a scalar; both would otherwise have been measured as candidates first.
6. A cold confirmation overstates a gain, and one boosted round can decide a generation. The committed kernel reads 77.82 µs on the first measurement after an idle period and 82.94 µs once settled, so a three-round confirmation read +4.94% for a pair that measures +2.47%. Warm-up pairs first, a prepared input directory per arm, alternating order, and quote the median of the paired per-round ratios.
7. Say which instrument a number came from. The loop measures the event span; the stage records measure Nsight kernel time. The `k_step` step the loop accepted improved the span and is slower in kernel time in the guard-free structure, which is how the two views disagreed until the pipeline settled it.
