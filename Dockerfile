# Dockerfile for GPU profiling with full ncu/nsys access
# Requires: docker run --privileged --gpus all

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package with uv
RUN uv venv && uv pip install -e ".[dev]"

# Copy test scripts
COPY script.py .

# Set up environment for profiling
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Enable GPU performance counters (needs --privileged at runtime)
# This is set in entrypoint since /proc is read-only at build time

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Enable GPU profiling permissions\n\
echo -1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true\n\
echo 0 > /proc/sys/kernel/kptr_restrict 2>/dev/null || true\n\
\n\
# Activate venv and run command\n\
source /app/.venv/bin/activate\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["mage", "demo"]
