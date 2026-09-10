# mage-001: kernel contracts

The [public field note](https://superposition.github.io/mage/experiments/mage-001/)
develops the ideas. This file specifies the experiment. See the
[measurement record](experiments/mage-001-validation.md) and [reproduction guide](guide.md).

## Shared conventions

Inputs and outputs are contiguous, row-major FP32. Files use little-endian
encoding. Shape products must be positive and fit in a signed 32-bit integer;
the CSR edge count may be zero. The host rejects incorrect file lengths,
non-finite input values, and invalid CSR adjacency before allocating GPU buffers.

The harness compares every output element with its PyTorch reference using
`rtol=1e-4, atol=1e-4`, and rejects non-finite outputs. PyTorch TF32 matmul is
disabled. The tested random inputs use a normal distribution with standard
deviation 0.25 and seed 20260910. Passing these cases is not a proof for every
FP32 input or shape.

| Operation | Manifest dimensions | Input files | Default dimensions |
| --- | --- | --- | --- |
| `matmul` | [M, N, K] | a: M×K; b: K×N | [1024, 1024, 1024] |
| `gelu` | [R, D] | a: R×D; b: D (bias) | [4096, 768] |
| `layernorm` | [R, D] | a: R×D; b: D (gamma); c: D (beta) | [4096, 768] |
| `triangle` | [N, C] | a, b: N×N×C | [128, 32] |
| `neighbor` | [N, F, E] | a: N×F; b: E (weights); CSR files below | [4096, 64, 65536] |

## Matrix multiplication

$$C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.$$

Output shape: M×N. The implementation uses 16×16 shared-memory tiles,
zero-filled boundary loads, and two block barriers per tile. All threads,
including threads outside the output boundary, reach both barriers.

## Bias and GELU

For every row and feature, set $z=x+b$, then evaluate

$$y=\tfrac12z\left(1+\tanh\left(\sqrt{2/\pi}\,(z+0.044715z^3)\right)\right).$$

The device uses an approximate tanh instruction. Reference:
`torch.nn.functional.gelu(x + bias, approximate="tanh")`.

## Layer normalization

For each row, $\mu=D^{-1}\sum_jx_j$ and
$v=D^{-1}\sum_j(x_j-\mu)^2$. Output:

$$y_j=\gamma_j(x_j-\mu)/\sqrt{v+10^{-5}}+\beta_j.$$

Variance is centered and uses divisor D. A 256-thread block reduces each row;
thread t writes columns t+256q. Blocks own disjoint rows. Reference:
`torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps=1e-5)`.

## Triangle contraction

$$O_{ijc}=\sum_{k=0}^{N-1}A_{ikc}B_{jkc}.$$

Output shape: N×N×C. Reference: `torch.einsum("ikc,jkc->ijc", a, b)`.
This primitive omits the projections, gating, normalization, and other structure
of a complete protein-model triangle module.

## Weighted neighbor aggregation

$$Y_{if}=\sum_{e=p_i}^{p_{i+1}-1}w_eX_{s_e f}.$$

`rowptr.bin` contains N+1 unsigned 32-bit pointers p; `indices.bin` contains
E unsigned 32-bit source indices s. Pointers must be monotone, start at zero,
and end at E. Source indices must lie in [0, N). Duplicate edges contribute
repeatedly. A row with no incoming edges produces zero. The PyTorch reference
uses gathered source features, weighting, and `index_add_` into a zero output.
