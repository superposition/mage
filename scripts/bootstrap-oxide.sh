#!/usr/bin/env bash
# Ubuntu 22.04 / WSL2 toolchain setup. Installs no Linux display driver.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo 'Run as root: sudo bash scripts/bootstrap-oxide.sh' >&2
  exit 1
fi
previous_cuda=$(readlink -f /usr/local/cuda || true)
install -d /usr/share/keyrings
if [ ! -s /usr/share/keyrings/mage-llvm.gpg ]; then
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/mage-llvm.gpg
fi
printf '%s\n' 'deb [signed-by=/usr/share/keyrings/mage-llvm.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main' > /etc/apt/sources.list.d/mage-llvm.list
# Ignore unrelated stale third-party sources for this invocation.
sources=$(mktemp -d /tmp/mage-apt.XXXXXX)
trap 'rm -rf -- "$sources"' EXIT
cp /etc/apt/sources.list.d/mage-llvm.list "$sources/"
cp /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list "$sources/"
apt_options=(-o "Dir::Etc::sourceparts=$sources")
apt-get "${apt_options[@]}" update
DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y --no-install-recommends cuda-toolkit-13-0 clang-21 llvm-21 libclang-21-dev libclang-cpp21-dev libclang-common-21-dev build-essential pkg-config ruby-full ruby-dev zlib1g-dev
if [ -n "$previous_cuda" ] && [ -d "$previous_cuda" ]; then
  ln -sfn "$previous_cuda" /usr/local/cuda
fi
