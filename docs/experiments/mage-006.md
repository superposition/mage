---
title: The load in flight, and the shape Triton uses
permalink: /experiments/mage-006/
eyebrow: "Field note 004 / Mathematics on a GPU"
description: The matrix multiply stopped waiting on its own tile loads, and the layer norm was rewritten in the shape Triton's kernel uses, read from its generated code. Two rewrites, their measured kernel time, the attempts that measured worse, and the gap that remains open.
math: true
---
The third field note stopped at 80.00 µs of GPU kernel time for the Rust matrix multiply and 10.05 µs for layer normalization, from 343.99 µs and 18.64 µs in the first comparison. This note records the stages that followed, one per kernel. The matrix multiply was **waiting on its own tile loads**: every K tile was copied from global to shared memory, then a barrier, then the multiply-adds, with nothing overlapping the copy. The layer norm was rewritten in the **shape of the kernel Triton emits** for the same operation, which was read by compiling that kernel offline and censusing its PTX.

The [first field note]({{ '/experiments/mage-001/' | relative_url }}) holds the original comparison, the [second]({{ '/experiments/mage-002/' | relative_url }}) the first rewrite, the [third]({{ '/experiments/mage-003/' | relative_url }}) the shared-read and residency work; the journal entry for this stage is [The load in flight, and the shape Triton uses](https://superposition.github.io/journal/the-load-in-flight-and-the-shape-triton-uses/).

## Four stages per kernel

| Stage | Matrix multiplication | LayerNorm | Retained in |
| --- | --- | --- | --- |
| original arrangement | 343.99 µs | 18.64 µs | `mage-001` |
| first rewrite (PR #42) | 141.70 µs | 11.03 µs | `mage-002`, `mage-003` |
| second rewrite (PR #43 / #44) | 80.00 µs | 10.05 µs | `mage-003` |
| third rewrite (PR #49 / #52) | 68.62 µs | 8.97 µs | `mage-006` |

All eight values are GPU kernel time from Nsight Systems captures: the sum of captured kernel durations over 100 iterations of the same FP32 shape. The two values in the last row are read from the `mage-006` namespace, which is the evidence for this note; the other five come from the namespaces named beside them. The LayerNorm value at the first rewrite is the warp kernel of PR #42 measured in the later capture session — the `mage-002` namespace records 11.09 µs for the same kernel — and the LayerNorm value at the third rewrite is 8.85 µs in the three other sessions that measured it, against 8.97 µs in the retained capture.

The spans around the call move with them: 326.72 µs → 144.41 µs → 82.60 µs → 74.09 µs for the matrix multiply, and 19.47 µs → 15.43 µs → 12.91 µs → 12.26 µs for layer normalization, each the mean of 300 warmed CUDA-event samples in three rounds.

## The matrix multiply was waiting for its own loads

The kernel that entered this stage gives every thread a 4 × 4 tile of outputs inside a 64 × 64 block tile, with a 64-deep step in the contraction. One shared buffer holds the transposed A tile (row stride 68) and one holds the B tile: 33792 bytes, 56 registers, 256 blocks. Each K tile is copied global → shared, the block synchronizes, and only then do the multiply-adds run. Nothing overlaps the copy, so the arithmetic idles while the tile it needs is still arriving.

**PR #49** double-buffers the K tiles and issues the copies asynchronously (`tiled_matmul_pipeline`):

- Two shared buffers hold one 64 × 32 A tile and one 32 × 64 B tile each — 33792 bytes in total, the same footprint as the single-buffered 64-deep kernel it replaces.
- The A tile is staged with **four-byte** copies (`cp_async_ca_4`), because its shared destination is scattered by the transpose; the B tile uses **sixteen-byte** copies (`cp_async_ca_16`).
- While a tile's multiply-adds run, the copy for the tile two steps ahead is in flight. The wait is `cp_async_wait_group(1)`, so one group may still be outstanding, and only the final tile waits for all groups.
- The register shape is unchanged: 64 × 64 block tiles, 4 × 4 outputs per thread.

The first pipeline version was **wrong**, and the timing harness would have reported a 69 µs "win" from it. Two defects, both found by the committed correctness check rather than by reading the code:

- the buffer offsets were computed in the wrong units (a count of `i32` elements divided by four), so buffer 1 was read from inside buffer 0;
- the final tile consumed buffer `tiles % 2` instead of `(tiles - 1) % 2`.

Both were fixed before any measurement was kept; the timing of the broken version is not evidence. After the fixes the 1024-cubed product matches the PyTorch FP32 reference with a maximum absolute error of 0.0, and the capture prices the kernel at **68.62 µs** of GPU kernel time and 74.09 µs around the call, against 80.00 µs and 82.60 µs for the kernel it replaces. The kernel time is stable across four sessions: 68.35, 68.60, 68.68 and 68.62 µs.

## The layer norm was rewritten in the shape Triton uses

Layer normalization moved for a different reason than the matrix multiply. Nothing here was waiting on a copy; the question was why Triton's kernel is faster at the same shape. Nsight Compute counters are unavailable on this host, so the comparison was made against the generated code instead. Triton's `norm_kernel` — the same configuration the harness runs, `BLOCK` 1024 and four warps — was compiled offline for `sm_89` through `ASTSource` and `GPUTarget`, with no GPU involved, and the PTX and cubin were censused:

| Property | Triton `norm_kernel` | committed `layer_norm_pair` (before this stage) |
| --- | --- | --- |
| Grid × block | 4096 × 128 (one block per row) | 1024 × 256 (two warps per row) |
| Registers | 31 (`cuobjdump`) | 39 |
| Shared memory | 16 bytes | 64 bytes |
| Loads | 24 scalar `ld.global.b32` covering x, scale and bias | 12 quad loads |
| Stores | 8 scalar | quad |
| Passes over x | **one** (the row slice stays in registers) | two |
| Barriers | 3 | 1 |

The lesson is not vectorisation: Triton's loads are scalar 32-bit, not 128-bit. It is **occupancy and a single pass** — 31 registers, almost no shared memory, one block per row, and the eight elements each thread owns staying in registers from the load to the store.

**PR #52** rebuilds the Rust kernel on that shape (`layer_norm_row`):

- grid = the number of rows (4096), one block of 128 threads per row;
- each thread holds a quad of the row in its first slot and a masked quad in its second, so the row is read **once** and the sum and the sum of squares are computed from the values already held;
- the four warps reduce with `shuffle_down` offsets and meet once in **eight floats of shared memory behind one barrier**;
- the output pass writes from the same registers, and the reciprocal square root is `rsqrt_approx_f32` — one instruction, about 1e-7 relative error, far inside the 1e-4 tolerance — instead of `1.0 / sqrt_rn_f32`, which is what Triton emits.

Wider rows keep the two-warp kernel and widths that are not a multiple of four keep the scalar one. The capture prices the new kernel at **8.97 µs** of GPU kernel time and 12.26 µs around the call, against 10.05 µs and 12.91 µs for the kernel it replaces.

## The layer norm is still behind Triton, and the gap is open

Triton's layer normalization has returned 7.99, 8.06, 8.07, 8.09, 8.10 and 8.32 µs of kernel time across the sessions in which it was captured; the Rust kernel measures 8.85 µs in three sessions and 8.97 µs in the retained one. The gap is **7–10%** and stable. The register profile now matches, the structure is the same single pass over the row, and what remains cannot be attributed on this host because Nsight Compute counters are unavailable. The gap is tracked in [mage issue #54](https://github.com/superposition/mage/issues/54) and is not treated as closed here.

## What the mage-006 captures show

The retained evidence is 15 Nsight Systems captures, one per operation and implementation, and three rounds of 100 warmed CUDA-event samples per implementation. PyTorch, Triton and the two rewritten Rust kernels were captured in the same session. The progression below is the centre of this note; four further views of the same five operations follow.

How each kernel reached its measured time:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-006/kernel-progression-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-006/kernel-progression.svg' | relative_url }}" width="740" height="510"
         alt="Two horizontal step charts of GPU kernel time in microseconds. Matrix multiplication falls from 343.99 microseconds at mage-001 to 141.70 with the 4 by 4 register tile, to 80.00 with the 128-bit shared reads and the 64-deep K step, and to 68.62 with the cp.async double buffer. LayerNorm falls from 18.64 to 11.03 with one warp per row, to 10.05 with two warps per row, and to 8.97 with one block per row holding a quad per thread. Each stage is labelled with the change that produced it.">
  </picture>
  <figcaption>
    <p>GPU kernel time at each stage, from separate Nsight Systems captures of 100 iterations on the same FP32 shapes. The factor printed between stages is that step's speed-up. Every stage value that also appears in a retained capture is read from it; the attempts that were measured and not adopted are listed separately, with their metric, in the table below.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-006/kernel-progression.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-006/kernel-progression.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/scripts/plot-kernel-progression.py">Script ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs of GPU kernel time)</summary>
      <table>
        <caption class="visually-hidden">GPU kernel time per stage for matrix multiplication and layer normalization</caption>
        <thead><tr><th scope="col">Stage</th><th scope="col">Matrix multiplication</th><th scope="col">LayerNorm</th><th scope="col">What changed at this stage</th></tr></thead>
        <tbody>
          <tr><th scope="row">mage-001</th><td>343.99</td><td>18.64</td><td>16 × 16 tiles, one output per thread; one 256-thread block per row with three passes and two tree reductions</td></tr>
          <tr><th scope="row">PR #42</th><td>141.70</td><td>11.03</td><td>4 × 4 outputs per thread, 64 × 64 block tiles, 128-bit global loads, K step 32; one warp per row with 128-bit quads and shuffle reductions</td></tr>
          <tr><th scope="row">PR #43 / #44</th><td>80.00</td><td>10.05</td><td>128-bit shared reads, the transposed A tile and a 64-deep K step; two warps per row with one shared exchange behind a single barrier</td></tr>
          <tr><th scope="row">PR #49 / #52</th><td>68.62</td><td>8.97</td><td>Two shared buffers with the next K tile loading as cp.async during the multiply-adds; one block per row with a quad per thread, a single pass over the row and eight floats of shared behind one barrier</td></tr>
        </tbody>
      </table>
      <p>Every value is GPU kernel time from a Nsight Systems capture. The LayerNorm value at PR #42 is the warp kernel as measured in the <code>mage-003</code> capture session; the <code>mage-002</code> namespace records 11.09 µs for the same kernel. The other six values are read from the namespace of the stage named beside them: 343.99 and 18.64 from <code>mage-001</code>, 141.70 from <code>mage-002</code>, 80.00 and 10.05 from <code>mage-003</code>, and 68.62 and 8.97 from <code>mage-006</code>.</p>
    </details>
  </figcaption>
</figure>
</div>

Kernel time and time around the call in one figure:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-006/comparison-views-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-006/comparison-views.svg' | relative_url }}" width="740" height="351"
         alt="GPU kernel time and time around the call per operation in the mage-006 capture: in the kernel panel PyTorch has the shortest time for matrix multiplication and triangle contraction, and Triton for Bias + GELU, LayerNorm and neighbor aggregation; in the span panel Rust has the shortest span for Bias + GELU, LayerNorm and neighbor aggregation, and PyTorch for matrix multiplication and triangle contraction.">
  </picture>
  <figcaption>
    <p>Top row: time inside the kernels. Bottom row: time around the call. Each column has its own scale, so implementations compare within a column.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-006/comparison-views.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-006/comparison-views.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-profiles.json">Captures ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs)</summary>
      <table>
        <caption class="visually-hidden">Rust kernel time and event span per operation in the mage-006 capture</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">Kernel</th><th scope="col">Span</th><th scope="col">Kernel, mage-001</th><th scope="col">Kernel, mage-003</th></tr></thead>
        <tbody>
          <tr><th scope="row">Matrix multiplication</th><td>68.6</td><td>74.1</td><td>344.0</td><td>80.0</td></tr>
          <tr><th scope="row">Bias + GELU</th><td>11.2</td><td>14.0</td><td>11.2</td><td>11.0</td></tr>
          <tr><th scope="row">LayerNorm</th><td>9.0</td><td>12.3</td><td>18.6</td><td>10.1</td></tr>
          <tr><th scope="row">Triangle contraction</th><td>81.6</td><td>86.9</td><td>81.2</td><td>80.1</td></tr>
          <tr><th scope="row">Neighbor aggregation</th><td>10.3</td><td>15.9</td><td>10.3</td><td>10.3</td></tr>
        </tbody>
      </table>
    </details>
  </figcaption>
</figure>
</div>

GPU kernel time for all five operations and all three implementations:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-006/comparison-kernel-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-006/comparison-kernel.svg' | relative_url }}" width="740" height="650"
         alt="GPU kernel time per operation in the mage-006 capture: PyTorch has the shortest time for matrix multiplication and triangle contraction, and Triton for Bias + GELU, LayerNorm and neighbor aggregation.">
  </picture>
  <figcaption>
    <p>GPU kernel time, summed per operation. Separate Nsight Systems capture, 100 iterations per measurement; gaps between launches are excluded. The WSL timestamp fallback has reduced precision. The Triton matrix-multiply capture in this session ran under load; see the control caveat below.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-006/comparison-kernel.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-006/comparison-kernel.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-profiles.json">Captures ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs)</summary>
      <table>
        <caption class="visually-hidden">GPU kernel time per operation and implementation</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">PyTorch</th><th scope="col">Triton</th><th scope="col">Rust</th></tr></thead>
        <tbody>
          <tr><th scope="row">Matrix multiplication</th><td>66.2</td><td>100.8</td><td>68.6</td></tr>
          <tr><th scope="row">Bias + GELU</th><td>18.3</td><td>7.7</td><td>11.2</td></tr>
          <tr><th scope="row">LayerNorm</th><td>11.2</td><td>8.3</td><td>9.0</td></tr>
          <tr><th scope="row">Triangle contraction</th><td>28.6</td><td>100.5</td><td>81.6</td></tr>
          <tr><th scope="row">Neighbor aggregation</th><td>120.9</td><td>6.1</td><td>10.3</td></tr>
        </tbody>
      </table>
      <p>The PyTorch matrix multiply is a cuBLAS library call and its kernel time has ranged from 44.03 to 66.22 µs across the four namespaces. Triton's matrix multiply is not stable between sessions and its single capture here is above the range it occupies elsewhere; see the limits.</p>
    </details>
  </figcaption>
</figure>
</div>

Time around the call:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-006/comparison-event-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-006/comparison-event.svg' | relative_url }}" width="740" height="650"
         alt="Time around the call: PyTorch has the shortest event span for matrix multiplication and triangle contraction, and Rust for Bias + GELU, LayerNorm and neighbor aggregation. These spans include possible launch gaps.">
  </picture>
  <figcaption>
    <p>Mean of 300 warmed CUDA-event spans, collected in three rounds with rotating implementation order. Whiskers show the range of the three round means, not a confidence interval. A span can include gaps while the host submits work.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-006/comparison-event.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-006/comparison-event.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-results.json">Samples ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values (µs)</summary>
      <table>
        <caption class="visually-hidden">Event span per operation and implementation</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">PyTorch</th><th scope="col">Triton</th><th scope="col">Rust</th></tr></thead>
        <tbody>
          <tr><th scope="row">Matrix multiplication</th><td>61.0</td><td>93.3</td><td>74.1</td></tr>
          <tr><th scope="row">Bias + GELU</th><td>35.6</td><td>28.4</td><td>14.0</td></tr>
          <tr><th scope="row">LayerNorm</th><td>26.4</td><td>27.0</td><td>12.3</td></tr>
          <tr><th scope="row">Triangle contraction</th><td>75.0</td><td>100.1</td><td>86.9</td></tr>
          <tr><th scope="row">Neighbor aggregation</th><td>124.1</td><td>27.6</td><td>15.9</td></tr>
        </tbody>
      </table>
      <p>The Rust matrix multiply moves from 82.60 µs around the call in <code>mage-003</code> to 74.09 µs here, and layer normalization from 12.91 µs to 12.26 µs.</p>
    </details>
  </figcaption>
</figure>
</div>

Kernel launches:

<div class="profile-comparison">
<figure class="profile-plot">
  <picture>
    <source media="(max-width: 520px)" srcset="{{ '/assets/figures/mage-006/comparison-launches-mobile.svg' | relative_url }}">
    <img src="{{ '/assets/figures/mage-006/comparison-launches.svg' | relative_url }}" width="740" height="650"
         alt="Kernel launches per operation: PyTorch launches 2 for Bias + GELU, 3 for triangle contraction and 4 for neighbor aggregation, and 1 for matrix multiplication and LayerNorm. Triton and Rust launch 1 for every operation.">
  </picture>
  <figcaption>
    <p>Captured launches divided by 100 iterations. Both custom implementations use one kernel per operation. Fewer launches explain part of the result; they do not determine kernel duration.</p>
    <div class="profile-links">
      <a href="{{ '/assets/figures/mage-006/comparison-launches.svg' | relative_url }}" download>Download SVG</a>
      <a href="{{ '/assets/figures/mage-006/comparison-launches.png' | relative_url }}" download>PNG</a>
      <a href="https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-profiles.json">Captures ↗</a>
    </div>
    <details class="profile-values">
      <summary>Read the plotted values</summary>
      <table>
        <caption class="visually-hidden">Kernel launches per operation and implementation</caption>
        <thead><tr><th scope="col">Operation</th><th scope="col">PyTorch</th><th scope="col">Triton</th><th scope="col">Rust</th></tr></thead>
        <tbody>
          <tr><th scope="row">Matrix multiplication</th><td>1</td><td>1</td><td>1</td></tr>
          <tr><th scope="row">Bias + GELU</th><td>2</td><td>1</td><td>1</td></tr>
          <tr><th scope="row">LayerNorm</th><td>1</td><td>1</td><td>1</td></tr>
          <tr><th scope="row">Triangle contraction</th><td>3</td><td>1</td><td>1</td></tr>
          <tr><th scope="row">Neighbor aggregation</th><td>4</td><td>1</td><td>1</td></tr>
        </tbody>
      </table>
    </details>
  </figcaption>
</figure>
</div>

The resource request the capture reports for the two rewritten kernels, against the kernels they replaced:

| Kernel | Grid × block | Registers / thread | Shared memory |
| --- | --- | --- | --- |
| `tiled_matmul_registers` (mage-003) | 256 × 256 threads | 56 | 33792 bytes |
| `tiled_matmul_pipeline` (mage-006) | 256 × 256 threads | 96 | 33792 bytes |
| `layer_norm_pair` (mage-003) | 1024 × 256 threads | 39 | 64 bytes |
| `layer_norm_row` (mage-006) | 4096 × 128 threads | 40 | 32 bytes |

The pipeline spends registers to hold the second buffer's indices and addresses; the layer norm's 32 bytes are the eight floats the four warps exchange behind one barrier.

## The control caveat

The same session that captured the two rewritten kernels also captured the Triton control for both. **The Triton matrix-multiply capture returned 100.80 µs, above the 71.94–90.09 µs band its other same-session captures occupy**, so that one capture ran under some load and its value is not a measurement of the Triton kernel on a quiet GPU; it is retained, labeled, and compared against the band rather than against a single number. The Triton layer-normalization control returned 8.32 µs, close to the 7.99–8.19 µs band it otherwise occupies, so the LayerNorm side of the comparison is usable. The two Rust kernels matched their stable ranges in the same session, which is why their values are quoted from it.

## Attempts that were measured and not adopted

Every entry is a **single-run development measurement**, not retained evidence, and the metric is named because the entries were not all taken with the same instrument.

| Attempt | Metric | Value | Compared with |
| --- | --- | --- | --- |
| LayerNorm, one block per row, 96 threads, `1/sqrt_rn` | GPU kernel time | 8.99 µs | 8.09 µs for Triton, same session |
| LayerNorm, one block per row, 192 threads holding one quad each | GPU kernel time | 9.11 µs | 8.85 µs for the 128-thread build |
| LayerNorm, four warps per row on an uneven split | CUDA-event span | 13.3–13.7 µs | 12.67 µs for the two-warp build, five runs |
| LayerNorm, row staged in shared memory for a single global pass | CUDA-event span | 14.3 µs | 12.67 µs for the two-warp build, five runs |
| LayerNorm, row held in registers on the two-warp split | CUDA-event span | 13.88 µs | 12.67 µs for the two-warp build, five runs |

The first two are the measurements that fixed the block size of the new kernel: 96 threads per row was slower than 128, and 192 threads with one quad each was slower again, so 128 threads holding a quad plus a masked tail slot is kept. The last three read the row in registers or in shared memory without changing the launch shape, and none of them beat the two-warp kernel they were compared against, so none of them was carried forward.

## The unchanged operations bound the drift

Bias + GELU, triangle contraction and neighbor aggregation keep their kernels. Between the `mage-003` and `mage-006` namespaces their GPU kernel time moves by 1.6%, 1.9% and 0.0%: 10.97 → 11.15 µs, 80.07 → 81.60 µs, and 10.33 → 10.33 µs. Those three numbers bound how much of the two rewritten kernels' movement could be environment drift rather than the rewrites. The library side is not as steady across namespaces — the PyTorch neighbor-aggregation kernel moves from 67.33 µs to 120.93 µs — so no baseline is compared across namespaces here.

## Limits

- **The library call is still ahead of the matrix multiply.** The Rust kernel is 68.62 µs of kernel time against 44.03–66.22 µs for the cuBLAS call behind PyTorch across the four namespaces, so the margin is small and session-dependent.
- **Triton's matrix multiply is not stable between sessions.** In the sessions where the two were captured together it has returned 71.94, 79.84, 86.95 and 90.09 µs, and 100.80 µs in the `mage-006` session, while the Rust kernel has moved only within 68.35–68.68 µs across four sessions. No ranking between the two follows from a single pairing, and the 100.80 µs capture is under load.
- **The layer-normalization gap to Triton is open.** The Rust kernel is 7–10% slower than Triton's (8.85–8.97 µs against 7.99–8.32 µs), with the register profile matched and the same single-pass structure. The difference cannot be attributed on this host, and it is tracked in [mage issue #54](https://github.com/superposition/mage/issues/54).
- **No counters.** Nsight Compute counters are unavailable on this host, so occupancy, cache-hit rates and memory throughput are not measured. The occupancy argument for both rewrites rests on the resource request each kernel makes, not on a measurement of what it held.
- **One capture per case.** Kernel time comes from a single Nsight Systems capture per implementation and operation and carries no interval of its own; the event spans are means of 300 samples over three rounds. The two views come from separate runs with different launch rhythms.
- **One workstation, unlocked clocks.** These are five fixed FP32 shapes on one WSL2 host with an RTX 4090 and unlocked clocks. Compilation, transfers, tile tails, lower precision, backward passes and end-to-end service behavior are outside the measurements.

## Method and reproduction

The harness is unchanged from the earlier notes: `examples/oxide/comparison.py` collects the CUDA-event spans, `profile_suite.py` captures the kernels with Nsight Systems, and `export_comparison.py` writes the retained evidence under `docs/assets/results/<experiment>` and rewrites each capture directory relative to the checkout. The run that produced this namespace wrote `artifacts/mage-006-final` and `artifacts/mage-006-nsys`. Those capture directories are local and not committed — `artifacts/` is ignored by git — so the evidence that travels with the repository is [comparison-results.json](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-results.json) and [comparison-profiles.json](https://github.com/superposition/mage/blob/master/docs/assets/results/mage-006/comparison-profiles.json), whose recorded capture paths are the relative `artifacts/mage-006-nsys/...` names.

```bash
.venv/bin/python examples/oxide/comparison.py --output artifacts/mage-006-final
.venv/bin/python examples/oxide/profile_suite.py \
  --inputs artifacts/mage-006-final --output artifacts/mage-006-nsys \
  --languages python triton rust
.venv/bin/python examples/oxide/export_comparison.py \
  --events artifacts/mage-006-final --profiles artifacts/mage-006-nsys --experiment mage-006
uv run --script scripts/plot-comparison.py --experiment mage-006
uv run --script scripts/plot-kernel-progression.py --experiment mage-006
```

The Triton census table comes from compiling `norm_kernel` in `examples/oxide/triton_target.py` offline, with the `ASTSource`/`GPUTarget` path and the `WIDTH` 768, `BLOCK` 1024, four-warp configuration the harness runs, and counting the loads, stores and barriers in the emitted PTX and the registers in the cubin with `cuobjdump`. The two plotting scripts read the committed evidence, validate the sample counts, correctness flags and launch counts, and document the stage values in this note; every stage value that exists in a committed namespace is asserted against it before the figure is written. The pages need only the committed files.

[Measurements and reproduction in GitHub](https://github.com/superposition/mage/blob/master/docs/experiments/mage-006.md) · [The third field note]({{ '/experiments/mage-003/' | relative_url }}) · [The open layer-normalization gap](https://github.com/superposition/mage/issues/54)
