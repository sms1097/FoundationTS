#!/usr/bin/env bash
set -euo pipefail

BASE_ARGS=(
  foundationts train
  --dataset-path time300b_selected
  --seq-max-len 4096
  --seq-stride 4096
  --num-expert-layers 1
  --hidden-size 384
  --n-head 12
  --n-decoder-layers 12
  --num-experts 8
  --k 2
  --d-ff 1536
  --d-expert 768
  --log-every 10
  --checkpoint-every 0
  --mfu-peak-tflops 989
  --max-wall-time-s 60
  --val-every 10 
  --final-val-on-budget
)

echo "Running baseline..."
"${BASE_ARGS[@]}" \
  --microbatch-size 32 \
  --global-batch-size 1024 \
  --run-name baseline

echo "Running patching high resolution..."
"${BASE_ARGS[@]}" \
  --microbatch-size 896 \
  --global-batch-size 1792 \
  --patch \
  --patch-len 32 \
  --patch-stride 16 \
  --run-name patch_32_16

echo "Running patching aggressive..."
"${BASE_ARGS[@]}" \
  --microbatch-size 2048 \
  --global-batch-size 8192 \
  --patch \
  --patch-len 64 \
  --patch-stride 64 \
  --run-name patch_64_64

echo "Running qk norm experiment..."
"${BASE_ARGS[@]}" \
  --microbatch-size 32 \
  --global-batch-size 1024 \
  --qk-norm \
  --run-name qk_norm

echo "Running sliding window attention (flash) experiment..."
"${BASE_ARGS[@]}" \
  --microbatch-size 32 \
  --global-batch-size 1024 \
  --attn-backend flash \
  --attn-window 256 \
  --run-name attn_window_256
