# Source this file in the shell used to build the cuTile example.
#
# cuTile Rust JIT-compiles kernels at launch, so the shell needs two things
# from the CUDA 13.3 tree: the toolkit root that `cuda-bindings` builds
# against, and the `tileiras` Tile IR assembler that `cutile-compiler` invokes
# (it first ships with CUDA 13.2; the toolkit here holds 13.3.36).
#
# Do not source this together with scripts/oxide-env.sh: that file pins the
# CUDA 13.0 tree for the cuda-oxide toolchain, and the two must not share a
# shell environment.
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export CUTILE_TILEIRAS_PATH="$CUDA_TOOLKIT_PATH/bin/tileiras"
export PATH="$CUDA_TOOLKIT_PATH/bin:$HOME/.cargo/bin:$PATH"
