# Build the Python and Rust experiment

## Keep both workflows

Mage already provides Triton kernels and autograd paths for operations including matrix multiplication, softmax, GELU, SiLU, RMSNorm, attention, RoPE, and fused MLP work. It also has CUDA-event timing for Triton, Nsight Systems and Nsight Compute backends, a terminal display, and SQLite history.

The Rust examples add an executable target to that profiling workflow. Existing Python imports and `mage profile` remain available.

## Prerequisites

The pinned cuda-oxide revision requires Linux, CUDA Toolkit 13 or later, a compatible NVIDIA driver (R580 or later for this setup), Clang/libclang 21, and its pinned Rust nightly. Use the Windows NVIDIA driver with WSL GPU support. Install the toolkit in WSL without a Linux display driver.

The bootstrap script is specific to Ubuntu 22.04. It expects NVIDIA's CUDA apt repository and keyring to be configured. It installs CUDA 13.0 and LLVM 21, preserving the prior `/usr/local/cuda` target.

```bash
git clone https://github.com/superposition/mage.git
cd mage
sudo bash scripts/bootstrap-oxide.sh
source scripts/oxide-env.sh

rustup toolchain install nightly-2026-08-28 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools,rustfmt
cargo +nightly-2026-08-28 install \
  --git https://github.com/NVlabs/cuda-oxide.git \
  --rev 26754ae52c26c097dc1c465a1e42c4c5d05a3d40 cargo-oxide
```

Install Mage into a separate Python environment. This investigation uses PyTorch 2.9.1 with its CUDA 12.8 wheel and Triton 3.5.1; that Python runtime can coexist with the CUDA 13 toolkit used to build Rust.

```bash
uv venv --python 3.13
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 triton==3.5.1
uv pip install -e . pytest
source .venv/bin/activate
```

## Build and check

```bash
source scripts/oxide-env.sh
cd examples/oxide
CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89
cd ../..
python examples/oxide/experiment.py --small --output artifacts/mage-001-small
python examples/oxide/experiment.py --output artifacts/mage-001
```

The first compiler build can take several minutes. The example's Cargo dependencies, compiler installation, and nightly are pinned together. Commit `Cargo.lock` when changing the toolchain.

The harness writes shared little-endian FP32 input files, SHA-256 hashes, a full-output correctness check against PyTorch, separate Python and Rust timing samples, and a combined result record. It fails when a result is non-finite or outside the documented tolerance.

## Then profile

On WSL, apply NVIDIA's documented Systems timestamp workaround before capturing:

```bash
source scripts/oxide-env.sh
python scripts/setup-nsys-wsl.py
```

This changes only `CuptiUseRawGpuTimestamps=false` in the configuration reported
by `nsys -z`. It uses a less precise time-conversion path that avoids missing
GPU events on the tested WSL setup. See the
[Nsight Systems 2025.3 release notes](https://docs.nvidia.com/nsight-systems/2025.3/ReleaseNotes/index.html).

Follow the [profiling guide](https://github.com/superposition/mage/blob/master/docs/profiling.md) for timing traces and hardware counters. Always collect ordinary event timings separately from an instrumented Nsight Compute run.

[Mathematical contracts](kernel-contracts.md) · [Measured evidence](experiments/mage-001-validation.md).
