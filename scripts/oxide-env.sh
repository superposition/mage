# Source this file in the shell used to build the examples.
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
export CUDA_OXIDE_LLC=/usr/bin/llc-21
export LIBCLANG_PATH=/usr/lib/llvm-21/lib
export PATH="$CUDA_TOOLKIT_PATH/bin:/usr/lib/llvm-21/bin:$HOME/.cargo/bin:$PATH"
