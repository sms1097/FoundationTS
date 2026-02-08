#!/usr/bin/env bash
set -e

VENV="$HOME/py310"
WHEEL_DIR="$HOME/wheels"
FA_VERSION="2.8.3"
TORCH_VERSION="2.8.0+cu128"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# -----------------------------
# Python 3.10 setup (once)
# -----------------------------
if [ ! -d "$VENV" ]; then
  echo ">>> Setting up Python 3.10 environment"
  sudo apt update
  sudo apt install -y python3.10 python3.10-venv python3.10-dev
  python3.10 -m venv "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Torch (stable + fast)
# -----------------------------
pip install -U "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
pip install -U psutil

# -----------------------------
# FlashAttention (H100)
# -----------------------------
mkdir -p "$WHEEL_DIR"

# Prefer binary wheels; fall back to source build if needed.
pip uninstall -y flash-attn || true
echo ">>> Installing FlashAttention (prefer binary)"
if ! pip install --only-binary=:all: "flash-attn==$FA_VERSION"; then
  echo ">>> No binary wheel found; building from source (H100 only)"
  export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
  export TORCH_CUDA_ARCH_LIST=90
  export FLASH_ATTENTION_CUDA_ARCHS=90
  export CMAKE_CUDA_ARCHITECTURES=90
  export CUDAARCHS=90
  export NVCC_FLAGS="-gencode arch=compute_90,code=sm_90"
  pip install --no-build-isolation --no-cache-dir "flash-attn==$FA_VERSION" -v
fi
pip install -e .

foundationts data download
